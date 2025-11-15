import os
import json
import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer

# ---------------------------
# 1. Detect language
# ---------------------------
def detect_language(text):
    try:
        lang = detect(text)
        return "ur" if lang == "ur" else "en"
    except:
        return "en"

# ---------------------------
# 2. Load FAISS index
# ---------------------------
def load_faiss_index(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"FAISS index not found: {path}")
    return faiss.read_index(path)

# ---------------------------
# 3. Load metadata
# ---------------------------
def load_metadata(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metadata file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ---------------------------
# 4. REAL embedding generator (offline)
# ---------------------------
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_real_embedding(text):
    return model.encode(text, convert_to_numpy=True).astype("float32")

# ---------------------------
# 5. Retrieve passages
# ---------------------------
def retrieve_passages(query_vec, index, metadata, top_k=4):
    query_vec = query_vec.reshape(1, -1).astype("float32")
    distances, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if idx == -1:
            continue
        results.append(metadata[idx])
    return results

# ---------------------------
# 6. Build context for testing (optional, for debugging)
# ---------------------------
def build_context(passages):
    ctx = ""
    for i, p in enumerate(passages):
        ctx += f"[Document {i+1}]\n{p['text']}\n\n"
    return ctx

# ---------------------------
# 7. Simplified answer (raw text from top passages)
# ---------------------------
def simple_answer(passages, char_limit=500):
    """
    Returns the text from the top passages without FAKE ANSWER boilerplate.
    """
    if not passages:
        return "⚠ No relevant documents found."

    combined_text = ""
    for p in passages:
        combined_text += p["text"] + "\n\n"

    # Limit output to char_limit
    return combined_text

# ---------------------------
# 8. Main RAG pipeline
# ---------------------------
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fyp_text", "faiss_indexes")

file_lang_map = {
    "en": "english",
    "ur": "urdu"
}

def rag_pipeline(question):
    print(f"\n🔎 Received Question: {question}")

    lang = detect_language(question)
    print(f"🌐 Detected Language: {lang}")

    file_prefix = file_lang_map.get(lang, "english")  # fallback to english

    index_path = os.path.join(BASE_DIR, f"{file_prefix}_faiss.index")
    meta_path = os.path.join(BASE_DIR, f"{file_prefix}_metadata.json")

    print(f"📁 Loading Index: {index_path}")
    print(f"📁 Loading Metadata: {meta_path}")

    index = load_faiss_index(index_path)
    print("FAISS index dimension:", index.d)
    metadata = load_metadata(meta_path)

    # REAL embedding instead of fake
    q_vec = get_real_embedding(question)

    print("🔍 Retrieving passages...")
    passages = retrieve_passages(q_vec, index, metadata)

    if not passages:
        return "⚠ No relevant documents found in FAISS."

    # ✅ Print retrieved chunks for debugging
    print("\n📄 Retrieved Chunks:")
    for i, p in enumerate(passages):
        print(f"--- Chunk {i+1} ---")
        print(json.dumps(p, indent=2, ensure_ascii=False))
        print("--------------------\n")

    # 🧠 Generate simplified answer from top passages
    print("🧠 Generating simplified answer...")
    answer = simple_answer(passages)

    return answer

# ---------------------------
# Run testing mode
# ---------------------------
if __name__ == "__main__":
    print("🚀 FREE RAG TESTING MODE (Offline, No API Required)\n")

    while True:
        q = input("Ask something (or 'exit'): ")
        if q.lower() == "exit":
            break

        print("\n" + rag_pipeline(q))
        print("\n" + "-"*80 + "\n")
