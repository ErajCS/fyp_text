import pdf2image
import pytesseract
from PIL import Image
import os

# Set the path to the Tesseract executable (update based on your system)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Directory containing the PDF files
pdf_dir = "C:\\Users\\Dell-5420\\Downloads\\fyp_github\\fyp_text"  # Update this path if needed

# List of specific filenames to process
target_files = [
    "batoor_adivsory_urdu.pdf"  # Adjust filename based on your PDF (e.g., match the uploaded image)
]

# Function to extract Urdu text using OCR
def extract_urdu_text(pdf_path):
    urdu_text = ""
    try:
        # Convert PDF pages to images
        images = pdf2image.convert_from_path(pdf_path)
        for image in images:
            custom_config = r'--oem 3 --psm 6 -l urd'
            text = pytesseract.image_to_string(image, config=custom_config)
            # Filter for Urdu text (Unicode range U+0600 to U+06FF)
            urdu_lines = [line for line in text.split('\n') if any(0x0600 <= ord(char) <= 0x06FF for char in line)]
            urdu_text += '\n'.join(urdu_lines) + '\n'
        return urdu_text
    except Exception as e:
        print(f"Error processing {pdf_path}: {e}")
        return ""

# Process each target file
for filename in os.listdir(pdf_dir):
    if any(target in filename for target in target_files):
        pdf_path = os.path.join(pdf_dir, filename)
        urdu_text = extract_urdu_text(pdf_path)
        
        # Save Urdu text to a file only if text is extracted
        if urdu_text.strip():
            output_file = os.path.join(pdf_dir, f"{os.path.splitext(filename)[0]}_urdu.txt")
            with open(output_file, "w", encoding="utf-8") as file:
                file.write(urdu_text)
            print(f"Extracted Urdu text saved to {output_file}")
        else:
            print(f"No Urdu text extracted from {filename}")