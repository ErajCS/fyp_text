# import re
# from pathlib import Path
# import logging
# import time
# import random
# from deep_translator import GoogleTranslator

# # ---------------- CONFIG ----------------
# ROOT_FOLDER = r"C:\Users\STAR PC\Desktop\fyp_text\hira_for_translation1"
# CHUNK_SIZE = 3000  # characters per chunk for safe translation
# # ----------------------------------------

# class TXTDeepTranslator:

#     def __init__(self, base_dir):
#         self.base_dir = Path(base_dir)
#         logging.basicConfig(
#             level=logging.INFO,
#             format="%(asctime)s - %(levelname)s - %(message)s",
#             handlers=[
#                 logging.FileHandler("deep_translation.log", encoding="utf-8"),
#                 logging.StreamHandler()
#             ]
#         )
#         self.logger = logging.getLogger(__name__)
#         self.stats = {"processed": 0, "skipped": 0, "failed": 0}

#     # ---------------------------
#     # CLEAN TEXT
#     # ---------------------------
#     def clean_text(self, text):
#         lines = []
#         for line in text.splitlines():
#             line = re.sub(r"\s+", " ", line).strip()
#             if len(line) > 1:
#                 lines.append(line)
#         return "\n".join(lines)

#     # ---------------------------
#     # LANGUAGE DETECTION
#     # ---------------------------
#     def detect_language(self, text):
#         ur = len(re.findall(r"[\u0600-\u06FF]", text))
#         en = len(re.findall(r"[A-Za-z]", text))
#         total = ur + en
#         if total == 0:
#             return "unknown"
#         if ur / total > 0.6:
#             return "urdu"
#         if en / total > 0.6:
#             return "english"
#         return "mixed"

#     # ---------------------------
#     # SAFE CHUNKED TRANSLATION
#     # ---------------------------
#     def translate_text(self, text, target_lang, chunk_size=CHUNK_SIZE, max_retries=5):
#         # Split text into chunks
#         chunks = []
#         start = 0
#         while start < len(text):
#             chunks.append(text[start:start+chunk_size])
#             start += chunk_size

#         translated_chunks = []

#         for i, chunk in enumerate(chunks):
#             for attempt in range(max_retries):
#                 try:
#                     translated = GoogleTranslator(
#                         source='auto',
#                         target='ur' if target_lang=='urdu' else 'en'
#                     ).translate(chunk)
#                     if target_lang == 'urdu':
#                         translated = f'\u202B{translated}\u202C'  # RTL markers
#                     translated_chunks.append(translated)
#                     break
#                 except Exception as e:
#                     wait_time = 2 + attempt*2 + random.random()
#                     self.logger.warning(f"⚠️ Chunk {i+1}/{len(chunks)} attempt {attempt+1} failed: {e}, retrying in {wait_time:.1f}s")
#                     time.sleep(wait_time)
#             else:
#                 # Only triggered if all retries fail
#                 self.logger.error(f"❌ Chunk {i+1} failed after {max_retries} attempts")
#                 translated_chunks.append("[Translation failed]")

#         return "\n".join(translated_chunks)

#     # ---------------------------
#     # PROCESS SINGLE TXT FILE
#     # ---------------------------
#     def process_txt(self, txt_path):
#         self.logger.info(f"📄 Processing {txt_path.name}")
#         try:
#             text = txt_path.read_text(encoding="utf-8", errors="ignore")
#             clean = self.clean_text(text)
#             if not clean:
#                 raise ValueError("Empty file")

#             lang = self.detect_language(clean)
#             self.logger.info(f"🧠 Detected language: {lang}")
#             base = txt_path.stem

#             if lang == "english":
#                 output_file = txt_path.parent / f"{base}_eng_urdu.txt"
#                 if output_file.exists():
#                     self.stats["skipped"] += 1
#                     self.logger.info("⏭️ Already translated, skipping")
#                     return
#                 translated = self.translate_text(clean, "urdu")
#                 output_file.write_text(translated, encoding="utf-8")

#             elif lang == "urdu":
#                 output_file = txt_path.parent / f"{base}_urdu_eng.txt"
#                 if output_file.exists():
#                     self.stats["skipped"] += 1
#                     self.logger.info("⏭️ Already translated, skipping")
#                     return
#                 translated = self.translate_text(clean, "english")
#                 # ---------------------------
#                 # Custom replacement: پکنک → PQNK
#                 # ---------------------------
#                 translated = translated.replace("Picnic", "PQNK").replace("picnic", "PQNK")
#                 output_file.write_text(translated, encoding="utf-8")

#             else:
#                 self.logger.warning("⚠️ Mixed language — skipping")
#                 self.stats["skipped"] += 1
#                 return

#             self.stats["processed"] += 1
#             self.logger.info("✅ Done")

#         except Exception as e:
#             self.stats["failed"] += 1
#             self.logger.error(f"❌ Failed {txt_path.name}: {e}")

#     # ---------------------------
#     # PROCESS ALL TXT FILES
#     # ---------------------------
#     def run(self):
#         txt_files = list(self.base_dir.rglob("*.txt"))
#         self.logger.info(f"🔍 Found {len(txt_files)} TXT files")
#         for txt in txt_files:
#             # Skip already translated files
#             if "_eng_urdu" in txt.name or "_urdu_eng" in txt.name:
#                 continue
#             self.process_txt(txt)
#         self.logger.info("📊 SUMMARY")
#         for k, v in self.stats.items():
#             self.logger.info(f"{k}: {v}")


# # ---------------------------
# # MAIN
# # ---------------------------
# def main():
#     print("🚀 TXT Translator (Deep Translator) with safe chunking & custom replacements")
#     processor = TXTDeepTranslator(ROOT_FOLDER)
#     processor.run()


# if __name__ == "__main__":
#     main()





import re
from pathlib import Path
import logging
import time
import random
from deep_translator import GoogleTranslator

# ---------------- CONFIG ----------------
ROOT_FOLDER = r"C:\\Users\\smwaj\\fyp_text\\\april_fina__videos_clean_transcripts"
CHUNK_SIZE = 3000  # characters per chunk for safe translation
# ----------------------------------------

class TXTDeepTranslator:

    def __init__(self, base_dir):
        self.base_dir = Path(base_dir)
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler("deep_translation.log", encoding="utf-8"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.stats = {"processed": 0, "skipped": 0, "failed": 0}

    # ---------------------------
    # CLEAN TEXT
    # ---------------------------
    def clean_text(self, text):
        lines = []
        for line in text.splitlines():
            line = re.sub(r"\s+", " ", line).strip()
            if len(line) > 1:
                lines.append(line)
        return "\n".join(lines)

    # ---------------------------
    # LANGUAGE DETECTION
    # ---------------------------
    def detect_language(self, text):
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
    # SAFE CHUNKED TRANSLATION
    # ---------------------------
    def translate_text(self, text, target_lang, chunk_size=CHUNK_SIZE, max_retries=5):
        chunks = []
        start = 0
        while start < len(text):
            chunks.append(text[start:start+chunk_size])
            start += chunk_size

        translated_chunks = []

        for i, chunk in enumerate(chunks):
            for attempt in range(max_retries):
                try:
                    translated = GoogleTranslator(
                        source='auto',
                        target='ur' if target_lang=='urdu' else 'en'
                    ).translate(chunk)
                    if target_lang == 'urdu':
                        translated = f'\u202B{translated}\u202C'
                    if translated is None:
                        self.logger.warning(f"⚠️ Chunk {i+1} returned None, retrying...")
                        raise ValueError("Translation returned None")
                    translated_chunks.append(translated)
                    break
                except Exception as e:
                    wait_time = 2 + attempt*2 + random.random()
                    self.logger.warning(f"⚠️ Chunk {i+1}/{len(chunks)} attempt {attempt+1} failed: {e}, retrying in {wait_time:.1f}s")
                    time.sleep(wait_time)
            else:
                self.logger.error(f"❌ Chunk {i+1} failed after {max_retries} attempts")
                translated_chunks.append("[Translation failed]")

        return "\n".join(translated_chunks)

    # ---------------------------
    # PROCESS SINGLE TXT FILE
    # ---------------------------
    def process_txt(self, txt_path):
        self.logger.info(f"📄 Processing {txt_path.name}")
        try:
            text = txt_path.read_text(encoding="utf-8", errors="ignore")
            clean = self.clean_text(text)
            if not clean:
                self.logger.warning(f"⚠️ Skipping {txt_path.name} (No speech detected in video)")
                self.stats["skipped"] += 1
                return

            lang = self.detect_language(clean)
            self.logger.info(f"🧠 Detected language: {lang}")

            original_base = txt_path.stem

            # ---------------------------
            # Rename original file with language suffix
            # ---------------------------
            if not original_base.endswith("_urdu") and not original_base.endswith("_Eng"):
                if lang == "urdu":
                    new_name = f"{original_base}_urdu.txt"
                elif lang == "english":
                    new_name = f"{original_base}_Eng.txt"
                else:
                    new_name = txt_path.name

                new_path = txt_path.parent / new_name
                txt_path.rename(new_path)
                txt_path = new_path
                self.logger.info(f"✏️ Renamed original file to {txt_path.name}")

            # Get clean base (without suffix)
            base_clean = original_base.replace("_urdu", "").replace("_Eng", "")

            if lang == "english":
                output_file = txt_path.parent / f"{base_clean}_urdu.txt"
                if output_file.exists():
                    self.stats["skipped"] += 1
                    self.logger.info("⏭️ Already translated, skipping")
                    return

                translated = self.translate_text(clean, "urdu")
                output_file.write_text(translated, encoding="utf-8")

            elif lang == "urdu":
                output_file = txt_path.parent / f"{base_clean}_Eng.txt"
                if output_file.exists():
                    self.stats["skipped"] += 1
                    self.logger.info("⏭️ Already translated, skipping")
                    return

                translated = self.translate_text(clean, "english")

                # Custom replacement
                translated = translated.replace("Picnic", "PQNK").replace("picnic", "PQNK")

                output_file.write_text(translated, encoding="utf-8")

            else:
                self.logger.warning("⚠️ Mixed language — skipping")
                self.stats["skipped"] += 1
                return

            self.stats["processed"] += 1
            self.logger.info("✅ Done")

        except Exception as e:
            self.stats["failed"] += 1
            self.logger.error(f"❌ Failed {txt_path.name}: {e}")

    # ---------------------------
    # PROCESS ALL TXT FILES
    # ---------------------------
    def run(self):
        txt_files = list(self.base_dir.rglob("*.txt"))
        self.logger.info(f"🔍 Found {len(txt_files)} TXT files")
        for txt in txt_files:
            if "_urdu" in txt.name or "_Eng" in txt.name:
                continue
            self.process_txt(txt)
        self.logger.info("📊 SUMMARY")
        for k, v in self.stats.items():
            self.logger.info(f"{k}: {v}")


# ---------------------------
# MAIN
# ---------------------------
def main():
    print("🚀 TXT Translator (Deep Translator) with safe chunking & custom replacements")
    processor = TXTDeepTranslator(ROOT_FOLDER)
    processor.run()


if __name__ == "__main__":
    main()