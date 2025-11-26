# import os
# import json
# import faiss
# import numpy as np
# from langdetect import detect
# from sentence_transformers import SentenceTransformer

# # ---------------------------
# # 1. Detect language
# # ---------------------------
# def detect_language(text):
#     try:
#         lang = detect(text)
#         return "ur" if lang == "ur" else "en"
#     except:
#         return "en"

# # ---------------------------
# # 2. Load FAISS index
# # ---------------------------
# def load_faiss_index(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"FAISS index not found: {path}")
#     return faiss.read_index(path)

# # ---------------------------
# # 3. Load metadata
# # ---------------------------
# def load_metadata(path):
#     if not os.path.exists(path):
#         raise FileNotFoundError(f"Metadata file not found: {path}")
#     with open(path, "r", encoding="utf-8") as f:
#         return json.load(f)

# # ---------------------------
# # 4. REAL embedding generator (offline)
# # ---------------------------
# model = SentenceTransformer("all-MiniLM-L6-v2")

# def get_real_embedding(text):
#     return model.encode(text, convert_to_numpy=True).astype("float32")

# # ---------------------------
# # 5. Retrieve passages
# # ---------------------------
# def retrieve_passages(query_vec, index, metadata, top_k=4):
#     query_vec = query_vec.reshape(1, -1).astype("float32")
#     distances, indices = index.search(query_vec, top_k)

#     results = []
#     for idx in indices[0]:
#         if idx == -1:
#             continue
#         results.append(metadata[idx])
#     return results

# # ---------------------------
# # 6. Build context for testing (optional, for debugging)
# # ---------------------------
# def build_context(passages):
#     ctx = ""
#     for i, p in enumerate(passages):
#         ctx += f"[Document {i+1}]\n{p['text']}\n\n"
#     return ctx

# # ---------------------------
# # 7. Simplified answer (raw text from top passages)
# # ---------------------------
# def simple_answer(passages, char_limit=500):
#     """
#     Returns the text from the top passages without FAKE ANSWER boilerplate.
#     """
#     if not passages:
#         return "⚠ No relevant documents found."

#     combined_text = ""
#     for p in passages:
#         combined_text += p["text"] + "\n\n"

#     # Limit output to char_limit
#     return combined_text

# # ---------------------------
# # 8. Main RAG pipeline
# # ---------------------------
# BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fyp_text", "faiss_indexes")

# file_lang_map = {
#     "en": "english",
#     "ur": "urdu"
# }

# def rag_pipeline(question):
#     print(f"\n🔎 Received Question: {question}")

#     lang = detect_language(question)
#     print(f"🌐 Detected Language: {lang}")

#     file_prefix = file_lang_map.get(lang, "english")  # fallback to english

#     index_path = os.path.join(BASE_DIR, f"{file_prefix}_faiss.index")
#     meta_path = os.path.join(BASE_DIR, f"{file_prefix}_metadata.json")

#     print(f"📁 Loading Index: {index_path}")
#     print(f"📁 Loading Metadata: {meta_path}")

#     index = load_faiss_index(index_path)
#     print("FAISS index dimension:", index.d)
#     metadata = load_metadata(meta_path)

#     # REAL embedding instead of fake
#     q_vec = get_real_embedding(question)

#     print("🔍 Retrieving passages...")
#     passages = retrieve_passages(q_vec, index, metadata)

#     if not passages:
#         return "⚠ No relevant documents found in FAISS."

#     # ✅ Print retrieved chunks for debugging
#     print("\n📄 Retrieved Chunks:")
#     for i, p in enumerate(passages):
#         print(f"--- Chunk {i+1} ---")
#         print(json.dumps(p, indent=2, ensure_ascii=False))
#         print("--------------------\n")

#     # 🧠 Generate simplified answer from top passages
#     print("🧠 Generating simplified answer...")
#     answer = simple_answer(passages)

#     return answer

# # ---------------------------
# # Run testing mode
# # ---------------------------
# if __name__ == "__main__":
#     print("🚀 FREE RAG TESTING MODE (Offline, No API Required)\n")

#     while True:
#         q = input("Ask something (or 'exit'): ")
#         if q.lower() == "exit":
#             break

#         print("\n" + rag_pipeline(q))
#         print("\n" + "-"*80 + "\n")











import os
import json
import faiss
import numpy as np
from langdetect import detect
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # <--- NEW IMPORT

# ---------------------------
# Setup OpenAI Client
# ---------------------------
# This will automatically look for the OPENAI_API_KEY environment variable.
# If you haven't set it in your system, you can pass api_key="sk-..." directly here (BUT BE CAREFUL!)
try:
    client = OpenAI()
except Exception as e:
    print(f"⚠️ Error initializing OpenAI: {e}")
    print("Did you set your OPENAI_API_KEY environment variable?")
    client = None

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
        # Load the list, but convert to a dict for faster lookup by ID
        data = json.load(f)
        return {item['id']: item for item in data}

# ---------------------------
# 4. REAL embedding generator (offline)
# ---------------------------
# MAKE SURE THIS MATCHES THE MODEL YOU USED TO CREATE THE DATABASE!
# If you used 'paraphrase-multilingual...', change this line!
# model = SentenceTransformer("all-MiniLM-L6-v2") 
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

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
        # Retrieve from the dict using the ID
        if idx in metadata:
            results.append(metadata[idx])
    return results

# ---------------------------
# 6. NEW: Generate GPT Answer
# ---------------------------
def generate_gpt_response(query, passages):
    """
    Sends the query and retrieved passages to GPT for a grounded answer.
    """
    if not client:
        return "⚠️ OpenAI client not connected. Check API Key."

    # 1. Build the Context String
    context_text = ""
    for p in passages:
        context_text += f"---\n{p['text']}\n"

    # 2. Create the Prompt
    # This instructs the model to only use your data.
    system_prompt = "You are a helpful assistant for a customized knowledge base. Answer the user question based ONLY on the provided context. If the answer is not in the context, say 'I don't have that information'."
    
    user_message = f"Context:\n{context_text}\n\nQuestion: {query}"

    print("🧠 Sending request to OpenAI...")

    try:
        # 3. Call the API (Using GPT-4o or GPT-3.5-turbo)
        response = client.chat.completions.create(
            model="gpt-4o",  # You can switch to "gpt-3.5-turbo" to save money
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3, # Low temp = more factual/focused
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error creating response: {e}"

# ---------------------------
# 7. Main RAG pipeline
# ---------------------------
# BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fyp_text", "faiss_indexes")
# We will use the absolute path to be 100% sure it finds the folder
# This points directly to: C:\Users\smwaj\fyp_text\faiss_indexes
BASE_DIR = r"C:\Users\smwaj\fyp_text\faiss_indexes"

file_lang_map = {
    "en": "english",
    "ur": "urdu"
}

def rag_pipeline(question):
    print(f"\n🔎 Received Question: {question}")

    lang = detect_language(question)
    print(f"🌐 Detected Language: {lang}")

    file_prefix = file_lang_map.get(lang, "english")

    index_path = os.path.join(BASE_DIR, f"{file_prefix}_faiss.index")
    meta_path = os.path.join(BASE_DIR, f"{file_prefix}_metadata.json")

    print(f"📁 Loading Index: {index_path}")
    
    # Load resources
    try:
        index = load_faiss_index(index_path)
        metadata = load_metadata(meta_path)
    except FileNotFoundError as e:
        return f"⚠️ Error: {e}"

    # Embed query
    q_vec = get_real_embedding(question)

    # Retrieve
    print("🔍 Retrieving passages...")
    passages = retrieve_passages(q_vec, index, metadata)

    if not passages:
        return "⚠ No relevant documents found in FAISS."

    # Print debug info
    print("\n📄 Retrieved Chunks (Context for GPT):")
    for i, p in enumerate(passages):
        print(f"--- Chunk {i+1} ---")
        # Printing just the first 100 chars to keep console clean
        print(p['text'][:100] + "...") 

    # Generate
    answer = generate_gpt_response(question, passages)

    return answer

# ---------------------------
# Run testing mode
# ---------------------------
if __name__ == "__main__":
    print("🚀 GPT-POWERED RAG TESTING MODE\n")

    while True:
        q = input("Ask something (or 'exit'): ")
        if q.lower() == "exit":
            break

        print("\n🤖 Answer:\n" + rag_pipeline(q))
        print("\n" + "-"*80 + "\n")















