import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
from langdetect import detect
from sentence_transformers import SentenceTransformer
from chardet import detect as chardet_detect

# --- CONFIG ---
BASE_DIR = r"C:\Users\Dell-5420\Downloads\fyp_github\fyp_text\text_pdfs"
CHUNK_SIZE = 500
OVERLAP = 100
BATCH_SIZE = 30
OUTPUT_DIR = "embeddings_output"

# --- Load multilingual model (supports Urdu + English) ---
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# --- Helpers ---
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(20000)
    result = chardet_detect(raw_data)
    return result['encoding'] if result['encoding'] else 'utf-8'

def detect_language_per_chunk(text):
    """Detect language more reliably per chunk."""
    urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[A-Za-z]', text))
    total_chars = len(text)

    if total_chars == 0:
        return 'unknown'

    urdu_ratio = urdu_chars / total_chars
    english_ratio = english_chars / total_chars

    # Dominant language
    if urdu_ratio > 0.3:       # at least 30% Urdu letters
        return 'ur'
    elif english_ratio > 0.3:  # at least 30% English letters
        return 'en'
    else:
        # fallback to langdetect for uncertain chunks
        try:
            lang = detect(text)
            return 'en' if 'en' in lang else 'ur' if 'ur' in lang else 'other'
        except:
            return 'unknown'

def chunk_text(text, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def clean_text(text):
    replacements = {
        "â€¢": "•", "â€“": "–", "â€”": "—", "â€˜": "‘", "â€™": "’",
        "â€œ": "“", "â€": "”", "â€¦": "…", "â€“": "-",
        "â†’": "→", "Â": "", "~": "~", "=": "=", "(": "(", ")": ")",
        "≈": "≈", "!": "!", ":": ":"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

# --- Batch Processing ---
os.makedirs(OUTPUT_DIR, exist_ok=True)

all_txt_files = []
for category in os.listdir(BASE_DIR):
    category_path = os.path.join(BASE_DIR, category)
    if not os.path.isdir(category_path):
        continue
    for file in os.listdir(category_path):
        if file.endswith(".txt"):
            all_txt_files.append((category, os.path.join(category_path, file)))

print(f"🔹 Total text files found: {len(all_txt_files)}")

batch_count = (len(all_txt_files) // BATCH_SIZE) + 1

for batch_num in range(batch_count):
    start_idx = batch_num * BATCH_SIZE
    end_idx = min((batch_num + 1) * BATCH_SIZE, len(all_txt_files))
    batch_files = all_txt_files[start_idx:end_idx]

    urdu_records, english_records = [], []

    print(f"\n⚙️ Processing batch {batch_num + 1}/{batch_count} "
          f"({len(batch_files)} files)...")

    for category, file_path in tqdm(batch_files, desc=f"Batch {batch_num+1}"):
        file = os.path.basename(file_path)
        encoding = detect_encoding(file_path)

        with open(file_path, "r", encoding=encoding, errors="ignore") as f:
            text = f.read().strip()

        text = clean_text(text)
        if len(text) < 50:
            continue

        chunks = chunk_text(text)

        for i, chunk in enumerate(chunks):
            lang = detect_language_per_chunk(chunk)
            if lang not in ['ur', 'en']:
                continue  # skip other or unknown languages

            embedding = model.encode(chunk)
            record = {
                "category": category,
                "filename": file,
                "chunk_id": i,
                "language": lang,
                "text": chunk
            }
            for j, val in enumerate(embedding):
                record[f"emb_{j}"] = val

            if lang == 'ur':
                urdu_records.append(record)
            elif lang == 'en':
                english_records.append(record)

    # --- Save Urdu ---
    if urdu_records:
        urdu_df = pd.DataFrame(urdu_records)
        urdu_csv = os.path.join(OUTPUT_DIR, f"urdu_embeddings_batch_{batch_num+1}.csv")
        urdu_df.to_csv(urdu_csv, index=False, encoding="utf-8-sig")

        urdu_vectors = urdu_df.filter(like='emb_').to_numpy()
        np.save(os.path.join(OUTPUT_DIR, f"urdu_vectors_batch_{batch_num+1}.npy"), urdu_vectors)
        print(f"Urdu batch {batch_num+1} saved ({len(urdu_df)} chunks)")

    # --- Save English ---
    if english_records:
        eng_df = pd.DataFrame(english_records)
        eng_csv = os.path.join(OUTPUT_DIR, f"english_embeddings_batch_{batch_num+1}.csv")
        eng_df.to_csv(eng_csv, index=False, encoding="utf-8-sig")

        eng_vectors = eng_df.filter(like='emb_').to_numpy()
        np.save(os.path.join(OUTPUT_DIR, f"english_vectors_batch_{batch_num+1}.npy"), eng_vectors)
        print(f"English batch {batch_num+1} saved ({len(eng_df)} chunks)")

    del urdu_records, english_records

print("\n🎉 All batches processed successfully!")
