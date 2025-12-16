import os
import pandas as pd
import numpy as np
import faiss
import json
import glob
from datetime import datetime
from urllib.parse import quote  # safely encode URLs

# --- Paths ---
MERGED_DIR = "embeddings_output/merged"
FAISS_DIR = "faiss_indexes"
DOCUMENTS_DIR = "text_pdfs"  # where PDFs are stored
os.makedirs(FAISS_DIR, exist_ok=True)

# --- Base URL where PDFs are hosted ---
BASE_URL = "https://yourdomain.com/pdfs"


def build_faiss_for_language(lang):
    print(f"\n🚀 Building FAISS index for {lang.capitalize()}...")

    # --- Find merged files automatically ---
    csv_files = glob.glob(os.path.join(MERGED_DIR, f"{lang}_embeddings_*.csv"))
    npy_files = glob.glob(os.path.join(MERGED_DIR, f"{lang}_vectors_*.npy"))

    if not csv_files or not npy_files:
        print(f"❌ No files found for {lang}. Check your merged folder.")
        return

    csv_path = csv_files[0]
    npy_path = npy_files[0]

    # --- Load data ---
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    vectors = np.load(npy_path).astype('float32')
    vectors = np.ascontiguousarray(vectors)
    faiss.normalize_L2(vectors)  # cosine similarity

    # --- Create and save FAISS index ---
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)

    index_path = os.path.join(FAISS_DIR, f"{lang}_faiss.index")
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index → {index_path}")

    # --- Create metadata JSON ---
    metadata = []
    current_time = datetime.utcnow().isoformat() + "Z"

    # Base PDF folder for each language
    pdf_base_url = BASE_URL if lang == "english" else f"{BASE_URL}/urdu_pdfs"

    for i, row in df.iterrows():
        filename = row.get("filename", "")

        # --- Convert .txt → .pdf properly ---
        if lang == "urdu":
            pdf_name = filename.replace("_urdu.txt", "_urdu.pdf")
        else:
            pdf_name = filename.replace(".txt", ".pdf")

        # --- Encode filename for valid URL ---
        pdf_url = f"{pdf_base_url}/{quote(pdf_name)}"

        record = {
            "id": int(i),
            "category": row.get("category", ""),
            "language": lang,
            "filename": pdf_name,
            "chunk_id": int(row.get("chunk_id", i)),
            "text": row.get("text", ""),
            "source_path": pdf_url,  # ✅ link to actual PDF
            "metadata": {
                "created_at": current_time,
                "source_type": "pdf"
            }
        }
        metadata.append(record)

    # --- Save metadata JSON ---
    json_path = os.path.join(FAISS_DIR, f"{lang}_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved metadata JSON → {json_path}")
    print(f"📦 Total records: {len(metadata)}")


# --- Run for both languages ---
build_faiss_for_language("english")
build_faiss_for_language("urdu")

print("\n🎉 FAISS indices + clean metadata JSON files with proper PDF URLs created successfully!")
