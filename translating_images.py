import base64
import logging
import time
from pathlib import Path
from openai import OpenAI
from deep_translator import GoogleTranslator

# ---------------- CONFIG ----------------
IMAGE_ROOT = r"C:\\Users\\smwaj\\fyp_text\\images"
SUPPORTED_EXTS = {".png", ".jpg", ".jpeg"}
MODEL = "gpt-4o-mini"
# ---------------------------------------

client = OpenAI(api_key="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")  

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("image_description.log", encoding="utf-8"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class ImageDescriber:
    def __init__(self, root_dir: str):
        self.root = Path(root_dir)
        self.stats = {"processed": 0, "skipped": 0, "failed": 0}

    # ---------------------------
    # IMAGE → BASE64
    # ---------------------------
    def image_to_base64(self, image_path: Path) -> str:
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    # ---------------------------
    # GPT IMAGE DESCRIPTION
    # ---------------------------
    def describe_image(self, image_b64: str, language: str) -> str:
        """
        language: 'urdu' or 'eng'
        """
        if language == "urdu":
            system_msg = (
                "آپ ایک ماہر ہیں جو اردو زبان میں زرعی تصاویر "
                "اور معلوماتی پوسٹرز کا تجزیہ کرتے ہیں۔"
            )
            user_text = (
                "اس تصویر کا مکمل تجزیہ اردو زبان میں کریں۔ "
                "بتائیں کہ تصویر کس بارے میں ہے، "
                "اس میں موجود اردو تحریر کا خلاصہ بیان کریں، "
                "اور تصویر کا مقصد واضح کریں۔"
            )
        else:  # English
            system_msg = (
                "You are an expert in analyzing images and informational posters."
            )
            user_text = (
                "Provide a detailed description of this image in English. "
                "Explain what it depicts, summarize any text present, "
                "and state the purpose of the image."
            )

        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": [
                    {"type": "text", "text": user_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                ]}
            ],
            max_tokens=600
        )
        return response.choices[0].message.content

    # ---------------------------
    # AUTOMATIC TRANSLATION
    # ---------------------------
    def translate_text(self, text: str, target_lang: str) -> str:
        """
        target_lang: 'en' or 'ur'
        """
        translator = GoogleTranslator(source="auto", target=target_lang)
        return translator.translate(text)

    # ---------------------------
    # PROCESS SINGLE IMAGE
    # ---------------------------
    def process_image(self, image_path: Path):
        # Determine original language from filename
        if "_urdu" in image_path.stem.lower():
            orig_lang = "urdu"
        elif "_eng" in image_path.stem.lower():
            orig_lang = "eng"
        else:
            logger.info(f"⏭️ Skipping (unknown language): {image_path.name}")
            self.stats["skipped"] += 1
            return

        # Define file paths
        eng_txt = image_path.with_name(image_path.stem.replace("_urdu", "_eng").replace("_eng", "_eng") + ".txt")
        urdu_txt = image_path.with_name(image_path.stem.replace("_eng", "_urdu").replace("_urdu", "_urdu") + ".txt")

        # Skip if BOTH files exist
        if eng_txt.exists() and urdu_txt.exists():
            logger.info(f"⏭️ Skipping (already done): {image_path.name}")
            self.stats["skipped"] += 1
            return

        try:
            logger.info(f"🖼️ Processing: {image_path}")
            image_b64 = self.image_to_base64(image_path)

            # Generate description in original language
            description_orig = self.describe_image(image_b64, orig_lang)

            if orig_lang == "urdu":
                # Save Urdu
                urdu_txt.write_text(description_orig, encoding="utf-8")
                # Translate to English
                description_eng = self.translate_text(description_orig, target_lang="en")
                eng_txt.write_text(description_eng, encoding="utf-8")
            else:  # English
                # Save English
                eng_txt.write_text(description_orig, encoding="utf-8")
                # Translate to Urdu
                description_urdu = self.translate_text(description_orig, target_lang="ur")
                urdu_txt.write_text(description_urdu, encoding="utf-8")

            self.stats["processed"] += 1
            time.sleep(2)  # avoid API throttling

        except Exception as e:
            logger.error(f"❌ Failed {image_path.name}: {e}")
            self.stats["failed"] += 1

    # ---------------------------
    # RUN ALL IMAGES
    # ---------------------------
    def run(self):
        images = [
            p for p in self.root.rglob("*")
            if p.suffix.lower() in SUPPORTED_EXTS
        ]

        logger.info(f"🔍 Found {len(images)} images to process")

        for img in images:
            self.process_image(img)

        logger.info("📊 SUMMARY")
        for k, v in self.stats.items():
            logger.info(f"{k}: {v}")


def main():
    print("🚀 GPT-4o-mini Image → Description Pipeline (Urdu + English)")
    processor = ImageDescriber(IMAGE_ROOT)
    processor.run()


if __name__ == "__main__":
    main()
