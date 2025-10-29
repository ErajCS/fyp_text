import os
import fitz  # PyMuPDF
import time
from deep_translator import GoogleTranslator

# Optional OCR fallback
try:
    from PIL import Image
    import pytesseract
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠️ pytesseract or PIL not installed — OCR fallback disabled. Run: pip install pytesseract pillow")

# 🔹 Base folder containing your Urdu PDFs
BASE_DIR = r"C:\Users\Dell-5420\Downloads\fyp_github\fyp_text"

# 🔸 Folders to skip
SKIP_FOLDERS = {'.git', '__pycache__', 'venv'}

# 📝 Specify exact PDF names you want to translate
# (Use exact file names as they appear in your folder)
TARGET_PDFS = {
    "Advisory_Paper_Mango_Pruning_Urdu.pdf"
    # Add more file names here...
}


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF using PyMuPDF, fallback to OCR if needed."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                page_text = page.get_text()
                if not page_text and OCR_AVAILABLE:
                    # OCR fallback for image-based PDFs
                    pix = page.get_pixmap()
                    img_path = "temp_page.png"
                    pix.save(img_path)
                    page_text = pytesseract.image_to_string(Image.open(img_path), lang="urd")  # Urdu OCR
                    os.remove(img_path)
                text += page_text
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text.strip()


def translate_large_text(text, chunk_size=3000, max_retries=3):
    """Translate long text into English with retry on connection errors."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []

    for i, chunk in enumerate(chunks, 1):
        success = False
        for attempt in range(1, max_retries + 1):
            try:
                eng_chunk = GoogleTranslator(source="ur", target="en").translate(chunk)
                translated_chunks.append(eng_chunk)
                print(f"   🔹 Translated chunk {i}/{len(chunks)} (try {attempt})")
                success = True
                time.sleep(1.5)
                break
            except Exception as e:
                print(f"   ⚠️ Error on chunk {i} (attempt {attempt}/{max_retries}): {e}")
                time.sleep(3)
        if not success:
            print(f"   ❌ Failed to translate chunk {i} after {max_retries} tries.")
            translated_chunks.append("[Translation failed for this section]")
    return "\n".join(translated_chunks)


# 🔸 Walk through all subfolders recursively
for root, dirs, files in os.walk(BASE_DIR):
    if any(skip in root for skip in SKIP_FOLDERS):
        continue

    category = os.path.basename(root)
    print(f"\n📂 Processing category: {category}")

    for file_name in files:
        if file_name not in TARGET_PDFS:  # Only process selected PDFs
            continue

        pdf_path = os.path.join(root, file_name)
        eng_txt_path = os.path.splitext(pdf_path)[0] + ".txt"

        # Skip already translated
        if os.path.exists(eng_txt_path):
            print(f"⏭️ Already translated: {file_name}")
            continue

        print(f"➡️ Extracting and translating: {file_name}")

        # Step 1: Extract Urdu text
        urdu_text = extract_text_from_pdf(pdf_path)
        if not urdu_text:
            print(f"⚠️ No text found in {file_name}")
            continue

        # Step 2: Translate to English
        english_text = translate_large_text(urdu_text)

        # Step 3: Save translation
        try:
            with open(eng_txt_path, "w", encoding="utf-8") as f:
                f.write(english_text)
            print(f"✅ English version saved: {eng_txt_path}")
        except Exception as e:
            print(f"❌ Error saving English file for {file_name}: {e}")
