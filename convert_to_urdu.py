import os
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
import time

# 🔹 Base folder containing category subfolders
BASE_DIR = r"C:\Users\STAR PC\Desktop\DS2 HW2\fyp_text"

def extract_text_from_pdf(pdf_path):
    """Extract text content from a PDF file using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text()
    except Exception as e:
        print(f"❌ Error reading {pdf_path}: {e}")
    return text.strip()

def translate_large_text(text, chunk_size=4000):
    """Translate long text into Urdu by splitting it into manageable chunks."""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    translated_chunks = []
    for i, chunk in enumerate(chunks, 1):
        try:
            urdu_chunk = GoogleTranslator(source="en", target="ur").translate(chunk)
            translated_chunks.append(urdu_chunk)
            print(f"   🔹 Translated chunk {i}/{len(chunks)}")
            time.sleep(1)  # avoid rate limit
        except Exception as e:
            print(f"   ❌ Error on chunk {i}: {e}")
            time.sleep(2)
    return "\n".join(translated_chunks)

# 🔸 Recursively go through all folders and process PDFs
for root, dirs, files in os.walk(BASE_DIR):
    category = os.path.basename(root)
    print(f"\n📂 Processing category: {category}")

    for file_name in files:
        if not file_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(root, file_name)
        urdu_txt_path = os.path.splitext(pdf_path)[0] + "_urdu.txt"

        # Skip if already translated
        if os.path.exists(urdu_txt_path):
            print(f"⏭️ Already translated: {file_name}")
            continue

        print(f"➡️ Extracting and translating: {file_name}")

        # Step 1: Extract text
        english_text = extract_text_from_pdf(pdf_path)
        if not english_text:
            print(f"⚠️ No text found in {file_name}")
            continue

        # Step 2: Translate to Urdu
        urdu_text = translate_large_text(english_text)

        # Step 3: Save Urdu translation
        with open(urdu_txt_path, "w", encoding="utf-8") as f:
            f.write(urdu_text)

        print(f"✅ Urdu version saved: {urdu_txt_path}")
