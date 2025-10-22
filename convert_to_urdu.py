import os
import fitz  # PyMuPDF
from deep_translator import GoogleTranslator
import time

# 🔹 Base folder containing category subfolders
BASE_DIR = r"C:\Users\STAR PC\Downloads\FYP\FYP\text_pdfs"

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

# 🔸 Loop through each category folder
for category in os.listdir(BASE_DIR):
    category_path = os.path.join(BASE_DIR, category)
    if not os.path.isdir(category_path):
        continue

    print(f"\n📂 Processing category: {category}")

    # Loop through PDF files in each category
    for file_name in os.listdir(category_path):
        if not file_name.lower().endswith(".pdf"):
            continue

        pdf_path = os.path.join(category_path, file_name)
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