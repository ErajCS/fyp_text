import os
import pytesseract
from PIL import Image
import cv2
import numpy as np
import re

# ---------------- CONFIG ----------------
IMAGES_ROOT = r"C:\\Users\\smwaj\\fyp_text\\images"
MIN_TEXT_LENGTH = 3
TEXT_RATIO_THRESHOLD = 0.8  # adjust if too many images are skipped

# Uncomment if needed
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# ---------------- UTILITIES ----------------

def preprocess_image(pil_img):
    """Convert image to binary for OCR"""
    gray = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

def is_mostly_text(pil_img):
    """Estimate if image is mostly text using black pixel ratio"""
    img = preprocess_image(pil_img)
    total_pixels = img.size
    black_pixels = np.sum(img == 0)
    ratio = black_pixels / total_pixels
    return ratio >= TEXT_RATIO_THRESHOLD

def detect_language(pil_img):
    """Detect whether image text is mostly English or Urdu"""
    img = preprocess_image(pil_img)
    text_eng = pytesseract.image_to_string(img, lang="eng").strip()
    text_urdu = pytesseract.image_to_string(img, lang="urd").strip()
    combined_text = text_eng + " " + text_urdu

    if len(combined_text) < MIN_TEXT_LENGTH:
        return None

    eng_chars = len(re.findall(r"[A-Za-z]", combined_text))
    urdu_chars = len(re.findall(r"[\u0600-\u06FF\u0750-\u077F]", combined_text))

    if eng_chars == 0 and urdu_chars == 0:
        return None
    return "urdu" if urdu_chars > eng_chars else "eng"

def rename_image(image_path, lang):
    """Rename image by appending _eng or _urdu if not already present"""
    folder, filename = os.path.split(image_path)
    name, ext = os.path.splitext(filename)

    # Skip if already renamed
    if name.lower().endswith("_eng") or name.lower().endswith("_urdu"):
        print(f"ℹ️ Already renamed: {filename}, skipping.")
        return

    new_name = f"{name}_{lang}{ext}"
    new_path = os.path.join(folder, new_name)
    os.rename(image_path, new_path)
    print(f"✅ Renamed: {filename} -> {new_name}")

# ---------------- MAIN ----------------

def main():
    for root, _, files in os.walk(IMAGES_ROOT):
        for file in files:
            if not file.lower().endswith((".jpg", ".jpeg", ".png")):
                continue

            image_path = os.path.join(root, file)

            try:
                pil_img = Image.open(image_path).convert("RGB")
            except Exception as e:
                print(f"⚠️ Failed to open {file}: {e}")
                continue

            # Optional: skip mostly-text images (uncomment if desired)
            # if is_mostly_text(pil_img):
            #     print(f"ℹ️ Skipping mostly text: {file}")
            #     continue

            lang = detect_language(pil_img)
            if lang in ["eng", "urdu"]:
                rename_image(image_path, lang)
            else:
                print(f"ℹ️ Language not detected: {file}")

    print("✅ Language detection & renaming completed.")

if __name__ == "__main__":
    main()