import os
import fitz  # PyMuPDF
#hello hira
# Path to your main folder
BASE_DIR = r"C:\Users\STAR PC\Downloads\FYP\FYP\text_pdfs"

for root, dirs, files in os.walk(BASE_DIR):
    for file in files:
        if file.lower().endswith(".pdf"):
            pdf_path = os.path.join(root, file)
            txt_path = os.path.splitext(pdf_path)[0] + ".txt"

            # Skip if txt already exists
            if os.path.exists(txt_path):
                print(f"⏭️ Skipping (already converted): {file}")
                continue

            text = ""
            with fitz.open(pdf_path) as doc:
                for page in doc:
                    text += page.get_text("text")

            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(text)

            print(f"✅ Converted: {file} → {os.path.basename(txt_path)}")

print("\n🎉 Conversion complete — only new PDFs processed!")