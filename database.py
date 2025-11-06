import os
import pandas as pd
import numpy as np
import faiss
import json
import glob

# --- Paths ---
MERGED_DIR = "embeddings_output/merged"
FAISS_DIR = "faiss_indexes"
os.makedirs(FAISS_DIR, exist_ok=True)

def build_faiss_for_language(lang):
    print(f"\n🚀 Building FAISS index for {lang.capitalize()}...")

    # --- Automatically find the merged CSV and NPY files ---
    csv_files = glob.glob(os.path.join(MERGED_DIR, f"{lang}_embeddings_*.csv"))
    npy_files = glob.glob(os.path.join(MERGED_DIR, f"{lang}_vectors_*.npy"))

    if not csv_files or not npy_files:
        print(f"❌ No files found for {lang}. Check your merged folder.")
        return

    csv_path = csv_files[0]  # pick the first match
    npy_path = npy_files[0]

    # --- Load data ---
    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    vectors = np.load(npy_path).astype('float32')

    # --- Ensure array is C-contiguous (FAISS requirement) ---
    vectors = np.ascontiguousarray(vectors)

    # --- Normalize for cosine similarity ---
    faiss.normalize_L2(vectors)

    # --- Create FAISS index ---
    dim = vectors.shape[1]
    index = faiss.IndexFlatIP(dim)  # Inner product = cosine similarity
    index.add(vectors)

    # --- Save index ---
    index_path = os.path.join(FAISS_DIR, f"{lang}_faiss.index")
    faiss.write_index(index, index_path)
    print(f"✅ Saved FAISS index → {index_path}")

    # --- Create JSON metadata file ---
    metadata = []
    for i, row in df.iterrows():
        metadata.append({
            "id": int(i),
            "category": row.get("category", ""),
            "filename": row.get("filename", ""),
            "chunk_id": row.get("chunk_id", ""),
            "text": row.get("text", "")
        })

    json_path = os.path.join(FAISS_DIR, f"{lang}_metadata.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved metadata JSON → {json_path}")
    print(f"📦 Total records: {len(metadata)}")


# --- Run for both languages ---
build_faiss_for_language("english")
build_faiss_for_language("urdu")

print("\n🎉 FAISS indices + metadata JSON files created successfully!")
