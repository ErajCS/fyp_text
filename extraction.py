import os
import re
import time
import random
import fitz  # PyMuPDF
from pathlib import Path
from google.cloud import vision
from deep_translator import GoogleTranslator
from langdetect import DetectorFactory
import logging

DetectorFactory.seed = 0

# ---------------- CONFIG ----------------
DPI = 300  # reduced for speed
ROOT_FOLDER = r"C:\Users\smwaj\fyp_text\text_pdfs"
# ---------------------------------------

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
except Exception:
    pass

class VisionPDFTranslator:
    def __init__(self, base_dir: str):
        self.base_dir = Path(base_dir)
        self.client = vision.ImageAnnotatorClient()


        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("vision_translation.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        self.stats = {"processed": 0, "skipped": 0, "failed": 0}

    # ---------------------------
    # LANGUAGE HINTS
    # ---------------------------
    def get_language_hints(self, filename: str):
        name = filename.lower()
        has_urdu = any(k in name for k in ["urdu", "اردو", "ur"])
        has_eng = any(k in name for k in ["english", "eng", "en"])

        if has_eng and has_urdu:
            return ["en", "ur"]
        if has_eng:
            return ["en"]
        if has_urdu:
            return ["ur"]
        return ["en", "ur"]

    # ---------------------------
    # OCR
    # ---------------------------
    def ocr_image(self, img_bytes: bytes, hints):
        image = vision.Image(content=img_bytes)
        response = self.client.document_text_detection(
            image=image,
            image_context={"language_hints": hints}
        )
        return response.full_text_annotation.text if response.full_text_annotation else ""

    def extract_pdf_text(self, pdf_path: Path):
        doc = fitz.open(pdf_path)
        texts = []

        hints = self.get_language_hints(pdf_path.name)
        self.logger.info(f"🔍 Language hints: {hints}")

        for page in doc:
            # 1. Try native text extraction first (Free, Fast, perfect for resumes)
            page_text = page.get_text().strip()
            
            if page_text:
                texts.append(page_text)
            else:
                # 2. Fallback to Google Cloud Vision OCR only if page is an image
                try:
                    pix = page.get_pixmap(dpi=DPI)
                    text = self.ocr_image(pix.tobytes("png"), hints)
                    texts.append(text.strip())
                except Exception as e:
                    self.logger.warning(f"⚠️ OCR skipped for page (Billing/API Error): {e}")

        return "\n".join(texts)

    # ---------------------------
    # CLEANING
    # ---------------------------
    def clean_text(self, text: str):
        lines = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) > 1:
                lines.append(line)
        return lines

    # ---------------------------
    # LANGUAGE DETECTION
    # ---------------------------
    def detect_language(self, text: str):
        ur = len(re.findall(r"[\u0600-\u06FF]", text))
        en = len(re.findall(r"[A-Za-z]", text))
        total = ur + en

        if total == 0:
            return "unknown"
        if ur / total > 0.6:
            return "urdu"
        if en / total > 0.6:
            return "english"
        return "mixed"

    # ---------------------------
    # CHUNKING
    # ---------------------------
    def chunk_lines(self, lines, max_chars=3500):
        chunks = []
        current = ""

        for line in lines:
            if len(current) + len(line) < max_chars:
                current += line + "\n"
            else:
                chunks.append(current.strip())
                current = line + "\n"

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def split_chunk(self, chunk):
        mid = len(chunk) // 2
        return chunk[:mid], chunk[mid:]

    # ---------------------------
    # TRANSLATION (LAST CHUNK FIXED)
    # ---------------------------
    def translate_lines(self, lines, target_lang):
        target = "ur" if target_lang == "urdu" else "en"
        chunks = self.chunk_lines(lines)
        output = []

        i = 0
        while i < len(chunks):
            chunk = chunks[i]
            is_last = (i == len(chunks) - 1)
            max_attempts = 4 if is_last else 2
            success = False

            for attempt in range(max_attempts):
                try:
                    translator = GoogleTranslator(source="auto", target=target)
                    translated = translator.translate(chunk)

                    if not translated or any(
                        bad in translated for bad in ["Error 504", "Server Error", "<html"]
                    ):
                        raise Exception("Server error detected")

                    if target_lang == "urdu":
                        translated = f"\u202B{translated}\u202C"

                    output.extend(translated.splitlines())
                    success = True
                    break

                except Exception as e:
                    self.logger.warning(
                        f"⚠️ Chunk {i+1}/{len(chunks)} attempt {attempt+1} failed"
                    )
                    time.sleep(5 + attempt * 3)

                    # 🔥 Special handling for LAST chunk
                    if is_last and attempt == 1 and len(chunk) > 1000:
                        self.logger.warning("🔁 Splitting last chunk and retrying")
                        a, b = self.split_chunk(chunk)
                        chunks[i] = a
                        chunks.insert(i + 1, b)
                        success = True
                        break

            if not success:
                output.append("[Translation failed]")

            sleep_time = 2 + random.uniform(0, 1)
            self.logger.info(f"⏳ Chunk {i+1}/{len(chunks)} done, sleeping {sleep_time:.1f}s")
            time.sleep(sleep_time)

            i += 1

        return output

    # ---------------------------
    # PROCESS SINGLE PDF
    # ---------------------------
    def process_pdf(self, pdf_path: Path):
        self.logger.info(f"📄 Processing {pdf_path.name}")

        base = pdf_path.stem
        urdu_file = pdf_path.parent / f"{base}_urdu.txt"
        eng_file = pdf_path.parent / f"{base}_english.txt"

        if urdu_file.exists() and eng_file.exists():
            self.logger.info("⏭️ Already processed")
            self.stats["skipped"] += 1
            return

        try:
            raw_text = self.extract_pdf_text(pdf_path)
            lines = self.clean_text(raw_text)

            if not lines:
                raise ValueError("No text extracted")

            lang = self.detect_language(" ".join(lines))
            self.logger.info(f"🧠 Detected language: {lang}")

            if lang == "english":
                eng_file.write_text("\n".join(lines), encoding="utf-8")
                urdu_lines = self.translate_lines(lines, "urdu")
                urdu_file.write_text("\n".join(urdu_lines), encoding="utf-8", errors="ignore")

            elif lang == "urdu":
                urdu_file.write_text(
                    "\n".join(f"\u202B{l}\u202C" for l in lines),
                    encoding="utf-8"
                )
                eng_lines = self.translate_lines(lines, "english")
                eng_file.write_text("\n".join(eng_lines), encoding="utf-8", errors="ignore")

            else:
                eng_file.write_text("\n".join(lines), encoding="utf-8")
                urdu_file.write_text(
                    "\n".join(f"\u202B{l}\u202C" for l in lines),
                    encoding="utf-8"
                )

            self.stats["processed"] += 1
            self.logger.info("✅ Done")

        except Exception as e:
            self.stats["failed"] += 1
            self.logger.error(f"❌ Failed {pdf_path.name}: {e}")

    # ---------------------------
    # PROCESS ALL
    # ---------------------------
    def run(self):
        pdfs = list(self.base_dir.rglob("*.pdf"))
        self.logger.info(f"🔍 Found {len(pdfs)} PDFs")

        for pdf in pdfs:
            self.process_pdf(pdf)

        self.logger.info("📊 SUMMARY")
        for k, v in self.stats.items():
            self.logger.info(f"{k}: {v}")


def main():
    print("🚀 Google Vision + Deep Translator (Last Chunk FIXED)")
    processor = VisionPDFTranslator(ROOT_FOLDER)
    processor.run()


if __name__ == "__main__":
    main()