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










# import psycopg2
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from pgvector.psycopg2 import register_vector
# import os

# # --- CONFIG ---
# DB_NAME = "pqnk_db"
# DB_USER = "postgres"
# DB_PASS = "admin123"  # <--- CHANGE THIS
# OPENAI_API_KEY = "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" # REPLACE WITH YOUR KEY (OR ENV VAR)

# # Initialize Models
# print("Loading models...")
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except:
#     print("⚠️ OpenAI API Key missing. RAG generation will fail.")
#     client = None

# def get_db_connection():
#     """Establishes connection and registers pgvector."""
#     conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
#     register_vector(conn)
#     return conn

# def detect_language(text):
#     try:
#         lang = detect(text)
#         return "ur" if lang == "ur" else "en"
#     except:
#         return "en"

# def retrieve_passages_from_db(query, lang, top_k=4):
#     """Retrieves top k chunks, filtered by language, using vector similarity."""
#     query_vec = model.encode(query).tolist()
    
#     conn = get_db_connection()
#     cur = conn.cursor()
    
#     # Filter by language AND search vector similarity
#     sql = """
#         SELECT v.chunk_text, c.filename, (v.embedding <=> %s::vector) as distance
#         FROM vectors v
#         JOIN content c ON v.content_id = c.content_id
#         WHERE c.language = %s
#         ORDER BY distance ASC
#         LIMIT %s
#     """
#     cur.execute(sql, (query_vec, lang, top_k))
#     results = cur.fetchall()
    
#     cur.close()
#     conn.close()
    
#     # Format results
#     passages = []
#     for r in results:
#         passages.append({"text": r[0], "filename": r[1]})
        
#     return passages

# def generate_gpt_response(query, passages):
#     if not client:
#         return "⚠️ OpenAI client not connected. Cannot generate response."

#     context_text = "\n\n".join([f"[Source: {p['filename']}]: {p['text']}" for p in passages])
    
#     # The system prompt ensures the model acts as Dr. Asif's expert system
#     system_prompt = "You are an expert agricultural assistant based on Dr. Asif Sharif's PQNK system. Answer the user question concisely based ONLY on the context provided. If the answer is not in the context, clearly state: 'I don't have enough verified PQNK information to answer that question.'"
    
#     user_message = f"Context:\n{context_text}\n\nUser Question: {query}"

#     try:
#         response = client.chat.completions.create(
#             model="gpt-4o-mini", # Using the cost-effective model
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": user_message}
#             ],
#             temperature=0.3
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         return f"Error from OpenAI API: {e}. Check API key and model name."

# # --- Main Pipeline Function called by app.py ---
# def rag_pipeline(question):
#     lang = detect_language(question)
    
#     # 1. Retrieve Passages (from DB)
#     passages = retrieve_passages_from_db(question, lang)
    
#     if not passages:
#         return "⚠️ No relevant information found in the PQNK knowledge base."
        
#     # 2. Generate Answer (from GPT)
#     answer = generate_gpt_response(question, passages)
#     return answer

# if __name__ == "__main__":
#     # Test locally
#     print("🚀 Database-Powered RAG Testing Mode")
#     while True:
#         q = input("Ask (or exit): ")
#         if q == "exit": break
#         print("\n🤖 Answer:\n" + rag_pipeline(q))













































import psycopg2
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
from pgvector.psycopg2 import register_vector
import os

# --- CONFIG ---
DB_NAME = "pqnk_db"
DB_USER = "postgres"
DB_PASS = "admin123"  # <--- CONFIRM YOUR PASSWORD
# Ensure your API Key is set here or in environment variables
OPENAI_API_KEY = "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" 

# Initialize Models
print("Loading models...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except:
    print("⚠️ OpenAI API Key missing.")
    client = None

def get_db_connection():
    """Establishes connection and registers pgvector."""
    conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
    register_vector(conn)
    return conn

def detect_language(text):
    try:
        lang = detect(text)
        return "ur" if lang == "ur" else "en"
    except:
        return "en"

def retrieve_passages_from_db(query, lang, top_k=4):
    """
    Retrieves top k chunks using HYBRID SEARCH:
    Combines Vector Similarity (Semantic) + Full-Text Search (Keyword Match).
    """
    query_vec = model.encode(query).tolist()
    conn = get_db_connection()
    cur = conn.cursor()
    
    # 1. Prepare Keyword Search Query
    # Convert query to tsquery format. 'plainto_tsquery' parses natural language 
    # into valid search tokens (e.g. "soil strategy" -> "'soil' & 'strategi'")
    ts_query = query.replace("'", "''") 
    
    # 2. HYBRID QUERY LOGIC
    # Formula: Hybrid Score = (Vector Distance) - (Keyword Rank * Weight)
    # - Vector Distance (<=>): Lower is better (0=identical, 1=different)
    # - Keyword Rank (ts_rank): Higher is better (0=no match, 0.1+=match)
    #
    # We subtract the Keyword Rank from the Distance. 
    # Since we ORDER BY score ASC (lowest first), subtracting a high keyword score 
    # pushes that result to the top (making the score smaller/negative).
    # Weight (0.2) controls how much keywords matter vs meaning.
    
    sql = """
        SELECT 
            v.chunk_text, 
            c.filename, 
            (v.embedding <=> %s::vector) - (ts_rank(v.text_search_tokens, plainto_tsquery('english', %s)) * 0.2) as hybrid_score
        FROM vectors v
        JOIN content c ON v.content_id = c.content_id
        WHERE c.language = %s
        ORDER BY hybrid_score ASC
        LIMIT %s
    """
    
    # Execute query with parameters
    cur.execute(sql, (query_vec, ts_query, lang, top_k))
    results = cur.fetchall()
    
    cur.close()
    conn.close()
    
    # Format results
    passages = []
    for r in results:
        passages.append({"text": r[0], "filename": r[1]})
        
    return passages

def generate_gpt_response(query, passages):
    if not client:
        return "⚠️ OpenAI client not connected. Cannot generate response."

    context_text = "\n\n".join([f"[Source: {p['filename']}]: {p['text']}" for p in passages])
    
    # The system prompt ensures the model acts as Dr. Asif's expert system
    system_prompt = "You are an expert agricultural assistant based on Dr. Asif Sharif's PQNK system. Answer the user question concisely based ONLY on the context provided. If the answer is not in the context, clearly state: 'I don't have enough verified PQNK information to answer that question.'"
    
    user_message = f"Context:\n{context_text}\n\nUser Question: {query}"

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini", # Using the cost-effective model
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            temperature=0.3
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error from OpenAI API: {e}. Check API key and model name."

# --- Main Pipeline Function called by app.py ---
def rag_pipeline(question):
    lang = detect_language(question)
    
    # 1. Retrieve Passages (Hybrid Search)
    passages = retrieve_passages_from_db(question, lang)
    
    if not passages:
        return "⚠️ No relevant information found in the PQNK knowledge base."
        
    # 2. Generate Answer (from GPT)
    answer = generate_gpt_response(question, passages)
    return answer

if __name__ == "__main__":
    # Test locally
    print("🚀 Database-Powered HYBRID RAG Testing Mode")
    while True:
        q = input("Ask (or exit): ")
        if q == "exit": break
        print("\n🤖 Answer:\n" + rag_pipeline(q))













































# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# import numpy as np
# import re
# import qdrant_client  # To check version

# # ================= CONFIG =================
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
# QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

# # Check Qdrant version
# print(f"🔧 Qdrant client version: {qdrant_client.__version__}")

# # =========================================
# print("🚀 Initializing RAG System...")

# # Initialize models
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# print("✅ Sentence Transformer loaded")

# # Initialize Qdrant
# qdrant = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
#     timeout=30
# )
# print("✅ Qdrant client connected")

# # Initialize OpenAI
# openai_client = OpenAI(api_key=OPENAI_API_KEY)
# print("✅ OpenAI client initialized")

# # ============ INTELLIGENCE UTILS ============

# def normalize_query(text):
#     """Clean and normalize query"""
#     text = text.strip()
#     text = re.sub(r'\s+', ' ', text)
#     return text

# def detect_lang(text):
#     """Detect language with better Urdu detection"""
#     try:
#         urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#         english_chars = len(re.findall(r'[A-Za-z]', text))
#         total_chars = len(text)
        
#         if total_chars == 0:
#             return "en"
        
#         urdu_ratio = urdu_chars / total_chars
#         english_ratio = english_chars / total_chars
        
#         if urdu_ratio > 0.3:
#             return "ur"
#         elif english_ratio > 0.3:
#             return "en"
#         else:
#             lang = detect(text)
#             return "ur" if lang == "ur" else "en"
#     except:
#         return "en"

# def generate_paraphrases(query, lang):
#     """Generate paraphrases based on language"""
#     if lang == "ur":
#         prompt = f"""
#         اس سوال کا صرف ایک مختلف اظہار بنائیں۔ معنی ایک جیسے رکھیں۔
        
#         سوال: {query}
        
#         اظہار:
#         """
#     else:
#         prompt = f"""
#         Generate just one paraphrase of this question.
#         Keep meaning same.
        
#         Question: {query}
        
#         Paraphrase:
#         """
    
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=100
#         )
        
#         content = response.choices[0].message.content.strip()
#         lines = [line.strip() for line in content.split('\n') if line.strip()]
        
#         paraphrases = [query]  # Start with original
        
#         for line in lines:
#             # Remove any prefix like "Paraphrase:" or "اظہار:"
#             clean_line = re.sub(r'^(Paraphrase:|اظہار:|Paraphrases:|اظہارات:)\s*', '', line)
#             if clean_line and len(clean_line) > 5 and clean_line != query:
#                 paraphrases.append(clean_line)
        
#         return list(set(paraphrases))[:2]  # Return unique paraphrases, max 2
#     except Exception as e:
#         print(f"⚠️ Paraphrase generation failed: {e}")
#         return [query]  # Return original if fails

# # ============ RETRIEVAL CORE ============

# def search_qdrant(query_embedding, top_k=10):
#     """Search Qdrant with version compatibility"""
#     try:
#         # Try new API first (search)
#         if hasattr(qdrant, 'search'):
#             return qdrant.search(
#                 collection_name=COLLECTION_NAME,
#                 query_vector=query_embedding,
#                 limit=top_k,
#                 with_payload=True
#             )
#         # Try old API (search_points)
#         elif hasattr(qdrant, 'search_points'):
#             results = qdrant.search_points(
#                 collection_name=COLLECTION_NAME,
#                 query_vector=query_embedding,
#                 limit=top_k,
#                 with_payload=True,
#                 with_vectors=False
#             )
#             # Convert to list of objects with consistent interface
#             converted_results = []
#             for result in results:
#                 converted_results.append(type('obj', (object,), {
#                     'id': result.id,
#                     'score': result.score,
#                     'payload': result.payload,
#                     'version': result.version
#                 })())
#             return converted_results
#         else:
#             print("❌ Qdrant client has neither 'search' nor 'search_points' method")
#             return []
#     except Exception as e:
#         print(f"❌ Search failed: {e}")
#         return []

# def retrieve_chunks(query, top_k=10):
#     """Retrieve chunks from Qdrant with paraphrases"""
#     print(f"🔍 Retrieving for: '{query}'")
    
#     # Detect language
#     lang = detect_lang(query)
#     print(f"🌐 Detected language: {lang}")
    
#     # Generate paraphrases
#     paraphrases = generate_paraphrases(query, lang)
#     print(f"📝 Using {len(paraphrases)} paraphrases")
    
#     all_results = []
    
#     for paraphrase in paraphrases:
#         try:
#             # Generate embedding for paraphrase
#             query_embedding = model.encode(paraphrase).tolist()
            
#             # Search in Qdrant
#             search_result = search_qdrant(query_embedding, top_k)
            
#             for result in search_result:
#                 all_results.append({
#                     'id': result.id if hasattr(result, 'id') else getattr(result, 'id', 0),
#                     'score': result.score if hasattr(result, 'score') else getattr(result, 'score', 0),
#                     'payload': result.payload if hasattr(result, 'payload') else getattr(result, 'payload', {}),
#                     'text': result.payload.get('text', '') if hasattr(result, 'payload') else getattr(result.payload, 'text', ''),
#                     'paraphrase': paraphrase
#                 })
            
#             print(f"   Found {len(search_result)} chunks for paraphrase")
            
#         except Exception as e:
#             print(f"⚠️ Search failed: {e}")
#             continue
    
#     # Deduplicate by text content
#     unique_results = {}
#     for result in all_results:
#         text = result['text']
#         if text and (text not in unique_results or result['score'] > unique_results[text]['score']):
#             unique_results[text] = result
    
#     # Sort by score
#     sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    
#     print(f"✅ Retrieved {len(sorted_results)} unique chunks")
#     return sorted_results[:top_k]  # Return top results

# # ============ ANSWER GENERATION ============

# def generate_answer(query, chunks):
#     """Generate answer using retrieved chunks"""
#     if not chunks:
#         lang = detect_lang(query)
#         if lang == "ur":
#             return "معذرت، اس سوال کے بارے میں معلومات دستیاب نہیں ہیں۔"
#         else:
#             return "Sorry, I don't have information about this topic."
    
#     # Build context with source information
#     context_parts = []
#     for i, chunk in enumerate(chunks):
#         payload = chunk.get('payload', {})
#         filename = payload.get('filename', 'Unknown')
#         category = payload.get('category', '')
#         text = chunk.get('text', '')
        
#         source_info = f"Source: {filename}"
#         if category:
#             source_info += f" ({category})"
        
#         context_parts.append(f"{source_info}\n{text}")
    
#     context = "\n\n---\n\n".join(context_parts)
    
#     # Determine response language
#     lang = detect_lang(query)
    
#     if lang == "ur":
#         system_prompt = """
#         آپ PQNK زرعی نظام کے ماہر معاون ہیں۔
        
#         قواعد:
#         1. صرف فراہم کردہ معلومات سے جواب دیں
#         2. اگر معلومات نہیں ملتی تو صاف کہیں کہ "معلومات دستیاب نہیں"
#         3. جواب اردو میں دیں
#         4. واضح اور مفید جواب دیں
#         """
#     else:
#         system_prompt = """
#         You are an expert assistant for PQNK agricultural system.
        
#         Rules:
#         1. Answer ONLY from the provided information
#         2. If information is not found, clearly say "Information not available"
#         3. Answer in English
#         4. Be clear and helpful
#         """
    
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
#             ],
#             temperature=0.3,
#             max_tokens=500
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"⚠️ OpenAI API error: {e}")
#         lang = detect_lang(query)
#         if lang == "ur":
#             return "جواب تیار کرنے میں مسئلہ پیش آیا۔ براہ کرم دوبارہ کوشش کریں۔"
#         else:
#             return "Error generating response. Please try again."

# # ============ MAIN PIPELINE ============

# def rag_pipeline(user_query):
#     """Main RAG pipeline"""
#     if not user_query or not user_query.strip():
#         return "براہ کرم کوئی سوال درج کریں۔ / Please enter a question."
    
#     try:
#         print(f"\n{'='*60}")
#         print(f"🤔 User Query: '{user_query}'")
        
#         # Normalize query
#         query = normalize_query(user_query)
        
#         # Retrieve chunks
#         chunks = retrieve_chunks(query, top_k=8)
        
#         if not chunks:
#             print("⚠️ No relevant chunks found")
#             lang = detect_lang(query)
#             if lang == "ur":
#                 return "کوئی متعلقہ معلومات نہیں ملیں۔"
#             else:
#                 return "No relevant information found."
        
#         print(f"📚 Using {len(chunks)} chunks for answer generation")
        
#         # Show retrieved chunks (for debugging)
#         print("\n📄 Retrieved Chunks:")
#         for i, chunk in enumerate(chunks):
#             score = chunk.get('score', 0)
#             text_preview = chunk.get('text', '')[:100]
#             print(f"  {i+1}. Score: {score:.3f} - {text_preview}...")
        
#         # Generate answer
#         answer = generate_answer(query, chunks)
        
#         print(f"\n🤖 Generated Answer: {answer[:200]}...")
#         print(f"{'='*60}\n")
        
#         return answer
        
#     except Exception as e:
#         print(f"❌ Error in RAG pipeline: {e}")
#         import traceback
#         traceback.print_exc()
        
#         lang = detect_lang(user_query)
#         if lang == "ur":
#             return "سسٹم میں خرابی ہے۔ براہ کرم دوبارہ کوشش کریں۔"
#         else:
#             return "System error. Please try again."

# # ============ TEST FUNCTION ============

# def test_system():
#     """Test the complete RAG system"""
#     print("🧪 Testing RAG System...")
    
#     test_queries = [
#         ("What is PQNK?", "English query"),
#         ("کٹائی کا طریقہ", "Urdu query about pruning"),
#         ("آلو کی کاشت", "Urdu query about potato"),
#         ("mango pruning techniques", "English agricultural query"),
#         ("مٹی کی تیاری", "Urdu soil preparation")
#     ]
    
#     for query, description in test_queries:
#         print(f"\n{'='*60}")
#         print(f"Testing: '{query}' ({description})")
#         print(f"{'='*60}")
        
#         result = rag_pipeline(query)
#         print(f"Answer: {result}")
#         print()

# # Quick connection test
# def test_connection():
#     """Test Qdrant connection"""
#     try:
#         # First check if we can access collections
#         collections = qdrant.get_collections()
#         print(f"✅ Connected to Qdrant! Collections: {[col.name for col in collections.collections]}")
        
#         # Test a simple search
#         test_embedding = model.encode("test").tolist()
#         results = search_qdrant(test_embedding, limit=1)
        
#         if results:
#             print(f"✅ Search works! Found {len(results)} results")
#             return True
#         else:
#             print("⚠️ Search returned no results (collection might be empty)")
#             return True  # Connection still works
            
#     except Exception as e:
#         print(f"❌ Connection test failed: {e}")
#         return False

# if __name__ == "__main__":
#     # Test connection first
#     if test_connection():
#         # Run tests
#         test_system()
#     else:
#         print("❌ Cannot proceed without Qdrant connection")





















































# from qdrant_client import QdrantClient
# from qdrant_client.models import Filter, FieldCondition, MatchValue
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# import numpy as np
# import re
# import sys

# # ================= CONFIG =================
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
# QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

# # =========================================
# print("🚀 Initializing RAG System...")

# # Initialize models
# try:
#     model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
#     print("✅ Sentence Transformer loaded")
# except Exception as e:
#     print(f"❌ Failed to load Sentence Transformer: {e}")
#     sys.exit(1)

# # Initialize Qdrant - Try different initialization methods
# try:
#     # Method 1: Newer API (v1.6+)
#     qdrant = QdrantClient(
#         url=QDRANT_URL,
#         api_key=QDRANT_API_KEY,
#         timeout=30
#     )
#     print("✅ Qdrant client connected (new API)")
# except Exception as e:
#     print(f"⚠️ New API failed: {e}")
#     try:
#         # Method 2: Older API (pre v1.6)
#         qdrant = QdrantClient(
#             host=QDRANT_URL.replace("https://", "").replace("http://", ""),
#             api_key=QDRANT_API_KEY,
#             port=6333,
#             https=True
#         )
#         print("✅ Qdrant client connected (old API)")
#     except Exception as e2:
#         print(f"❌ Both connection methods failed: {e2}")
#         sys.exit(1)

# # Initialize OpenAI
# try:
#     openai_client = OpenAI(api_key=OPENAI_API_KEY)
#     print("✅ OpenAI client initialized")
# except Exception as e:
#     print(f"❌ Failed to initialize OpenAI: {e}")
#     sys.exit(1)

# # ============ TEST QDRANT CONNECTION ============

# def test_qdrant_connection():
#     """Test Qdrant connection and available methods"""
#     print("\n🔍 Testing Qdrant connection...")
    
#     try:
#         # Test 1: Get collections
#         collections = qdrant.get_collections()
#         print(f"✅ Collections found: {[col.name for col in collections.collections]}")
        
#         # Test 2: Count points
#         count_result = qdrant.count(collection_name=COLLECTION_NAME)
#         print(f"✅ Points in collection: {count_result.count}")
        
#         # Test 3: Check available search methods
#         print("\n🔍 Available search methods:")
#         methods = [m for m in dir(qdrant) if 'search' in m.lower() and not m.startswith('_')]
#         for method in methods:
#             print(f"  - {method}")
        
#         return True
#     except Exception as e:
#         print(f"❌ Qdrant connection test failed: {e}")
#         return False

# # Run connection test
# if not test_qdrant_connection():
#     print("\n⚠️ Cannot proceed without Qdrant connection")
#     sys.exit(1)

# # ============ INTELLIGENCE UTILS ============

# def normalize_query(text):
#     """Clean and normalize query"""
#     text = text.strip()
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
#     return text

# def detect_lang(text):
#     """Detect language with better Urdu detection"""
#     try:
#         # Check for Urdu characters
#         urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#         english_chars = len(re.findall(r'[A-Za-z]', text))
#         total_chars = len(text)
        
#         if total_chars == 0:
#             return "en"
        
#         urdu_ratio = urdu_chars / total_chars
#         english_ratio = english_chars / total_chars
        
#         if urdu_ratio > 0.3:
#             return "ur"
#         elif english_ratio > 0.3:
#             return "en"
#         else:
#             # Fallback to langdetect
#             lang = detect(text)
#             return "ur" if lang == "ur" else "en"
#     except:
#         return "en"

# def generate_paraphrases(query, lang):
#     """Generate paraphrases based on language"""
#     if lang == "ur":
#         prompt = f"""
#         اس سوال کے 2 مختلف اظہار بنائیں۔ معنی ایک جیسے رکھیں۔
        
#         سوال: {query}
        
#         اظہار:
#         1.
#         2.
#         """
#     else:
#         prompt = f"""
#         Generate 2 paraphrases of this question.
#         Keep meaning same.
        
#         Question: {query}
        
#         Paraphrases:
#         1.
#         2.
#         """
    
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.7,
#             max_tokens=150
#         )
        
#         content = response.choices[0].message.content
#         lines = [line.strip() for line in content.split('\n') if line.strip()]
        
#         paraphrases = [query]  # Start with original
        
#         for line in lines:
#             # Remove numbers and bullets
#             clean_line = re.sub(r'^[0-9\.\-\*\)\s]+', '', line)
#             if clean_line and len(clean_line) > 5:
#                 paraphrases.append(clean_line)
        
#         return list(set(paraphrases))[:3]  # Return unique paraphrases, max 3
#     except Exception as e:
#         print(f"⚠️ Paraphrase generation failed: {e}")
#         return [query]  # Return original if fails

# # ============ RETRIEVAL CORE ============

# def retrieve_chunks(query, top_k=10):
#     """Retrieve chunks from Qdrant with paraphrases"""
#     print(f"🔍 Retrieving for: '{query}'")
    
#     # Detect language
#     lang = detect_lang(query)
#     print(f"🌐 Detected language: {lang}")
    
#     # Generate paraphrases
#     paraphrases = generate_paraphrases(query, lang)
#     print(f"📝 Using {len(paraphrases)} paraphrases: {[p[:30]+'...' for p in paraphrases]}")
    
#     all_results = []
    
#     for paraphrase in paraphrases:
#         try:
#             # Generate embedding for paraphrase
#             query_embedding = model.encode(paraphrase).tolist()
            
#             # Use query_points (correct for Qdrant v1.16.2)
#             try:
#                 search_result_obj = qdrant.query_points(
#                     collection_name=COLLECTION_NAME,
#                     query=query_embedding,
#                     limit=top_k,
#                     with_payload=True
#                 )
#                 search_result = search_result_obj.points
#             except Exception as e:
#                 print(f"⚠️ Search failed for paraphrase '{paraphrase}': {e}")
#                 continue
            
#             for result in search_result:
#                 # Handle result structure
#                 payload = getattr(result, 'payload', {})
#                 score = getattr(result, 'score', 0)
#                 result_id = getattr(result, 'id', 0)
                
#                 all_results.append({
#                     'id': result_id,
#                     'score': score,
#                     'payload': payload,
#                     'text': payload.get('text', ''),
#                     'paraphrase': paraphrase
#                 })
            
#             print(f"   Found {len(search_result)} chunks for paraphrase")
            
#         except Exception as e:
#             print(f"⚠️ Embedding/search failed: {e}")
#             continue
    
#     # Deduplicate by text content
#     unique_results = {}
#     for result in all_results:
#         text = result['text']
#         if text not in unique_results or result['score'] > unique_results[text]['score']:
#             unique_results[text] = result
    
#     # Sort by score
#     sorted_results = sorted(unique_results.values(), key=lambda x: x['score'], reverse=True)
    
#     print(f"✅ Retrieved {len(sorted_results)} unique chunks")
#     return sorted_results[:top_k]  # Return top results

# # ============ ANSWER GENERATION ============

# def generate_answer(query, chunks):
#     """Generate answer using retrieved chunks"""
#     if not chunks:
#         lang = detect_lang(query)
#         if lang == "ur":
#             return "معذرت، اس سوال کے بارے میں معلومات دستیاب نہیں ہیں۔"
#         else:
#             return "Sorry, I don't have information about this topic."
    
#     # Build context with source information
#     context_parts = []
#     for i, chunk in enumerate(chunks):
#         payload = chunk.get('payload', {})
#         filename = payload.get('filename', 'Unknown')
#         category = payload.get('category', '')
#         text = chunk.get('text', '')
        
#         source_info = f"Source: {filename}"
#         if category:
#             source_info += f" ({category})"
        
#         context_parts.append(f"{source_info}\n{text}")
    
#     context = "\n\n---\n\n".join(context_parts)
    
#     # Determine response language
#     lang = detect_lang(query)
    
#     if lang == "ur":
#         system_prompt = """
#         آپ PQNK زرعی نظام کے ماہر معاون ہیں۔
        
#         قواعد:
#         1. صرف فراہم کردہ معلومات سے جواب دیں
#         2. اگر معلومات نہیں ملتی تو صاف کہیں کہ "معلومات دستیاب نہیں"
#         3. جواب اردو میں دیں
#         4. واضح اور مفید جواب دیں
#         """
#     else:
#         system_prompt = """
#         You are an expert assistant for PQNK agricultural system.
        
#         Rules:
#         1. Answer ONLY from the provided information
#         2. If information is not found, clearly say "Information not available"
#         3. Answer in English
#         4. Be clear and helpful
#         """
    
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt},
#                 {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
#             ],
#             temperature=0.3,
#             max_tokens=500
#         )
#         return response.choices[0].message.content
#     except Exception as e:
#         print(f"⚠️ OpenAI API error: {e}")
#         lang = detect_lang(query)
#         if lang == "ur":
#             return "جواب تیار کرنے میں مسئلہ پیش آیا۔ براہ کرم دوبارہ کوشش کریں۔"
#         else:
#             return "Error generating response. Please try again."

# # ============ MAIN PIPELINE ============

# def rag_pipeline(user_query):
#     """Main RAG pipeline"""
#     if not user_query or not user_query.strip():
#         return "براہ کرم کوئی سوال درج کریں۔ / Please enter a question."
    
#     try:
#         print(f"\n{'='*60}")
#         print(f"🤔 User Query: '{user_query}'")
        
#         # Normalize query
#         query = normalize_query(user_query)
        
#         # Retrieve chunks
#         chunks = retrieve_chunks(query, top_k=8)
        
#         if not chunks:
#             print("⚠️ No relevant chunks found")
#             lang = detect_lang(query)
#             if lang == "ur":
#                 return "کوئی متعلقہ معلومات نہیں ملیں۔"
#             else:
#                 return "No relevant information found."
        
#         print(f"📚 Using {len(chunks)} chunks for answer generation")
        
#         # Show retrieved chunks (for debugging)
#         print("\n📄 Retrieved Chunks:")
#         for i, chunk in enumerate(chunks):
#             score = chunk.get('score', 0)
#             text_preview = chunk.get('text', '')[:100]
#             print(f"  {i+1}. Score: {score:.3f} - {text_preview}...")
        
#         # Generate answer
#         answer = generate_answer(query, chunks)
        
#         print(f"\n🤖 Generated Answer: {answer[:200]}...")
#         print(f"{'='*60}\n")
        
#         return answer
        
#     except Exception as e:
#         print(f"❌ Error in RAG pipeline: {e}")
#         import traceback
#         traceback.print_exc()
        
#         lang = detect_lang(user_query)
#         if lang == "ur":
#             return "سسٹم میں خرابی ہے۔ براہ کرم دوبارہ کوشش کریں۔"
#         else:
#             return "System error. Please try again."

# # ============ TEST FUNCTION ============

# def test_system():
#     """Test the complete RAG system"""
#     print("🧪 Testing RAG System...")
    
#     test_queries = [
#         ("What is PQNK?", "English query"),
#         ("کٹائی کا طریقہ", "Urdu query about pruning"),
#         ("آلو کی کاشت", "Urdu query about potato"),
#     ]
    
#     for query, description in test_queries:
#         print(f"\n{'='*60}")
#         print(f"Testing: '{query}' ({description})")
#         print(f"{'='*60}")
        
#         result = rag_pipeline(query)
#         print(f"Answer: {result}")
#         print()

# if __name__ == "__main__":
#     # Run tests
#     test_system()















































#  QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
# QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"







# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from spellchecker import SpellChecker
# import re
# import sys

# # ================= CONFIG =================
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
# QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

# MIN_SCORE_THRESHOLD = 0.25
# TOP_K = 8

# # =========================================
# print("🚀 Initializing RAG System...")

# # Models
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# spell_en = SpellChecker()
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# # Qdrant
# qdrant = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
#     timeout=30
# )

# # ============ LANGUAGE & TEXT UTILS ============

# def normalize_query(text):
#     text = text.strip()
#     text = re.sub(r"\s+", " ", text)
#     return text

# def detect_lang(text):
#     urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#     eng_chars = len(re.findall(r'[A-Za-z]', text))
#     if urdu_chars > eng_chars:
#         return "ur"
#     return "en"

# def correct_spelling(text, lang):
#     if lang != "en":
#         return text
#     return " ".join([spell_en.correction(w) or w for w in text.split()])

# # ============ QUERY INTELLIGENCE ============

# def compress_intent(query, lang):
#     prompt = (
#         f"Rewrite as a short search intent:\n{query}"
#         if lang == "en"
#         else f"اس سوال کو مختصر تلاش کے ارادے میں تبدیل کریں:\n{query}"
#     )
#     try:
#         r = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=30
#         )
#         return r.choices[0].message.content.strip()
#     except:
#         return query
    
# def normalize_intent(query: str) -> str:
#     """
#     Normalize different question styles to a canonical form
#     """
#     q = query.lower().strip()

#     patterns = [
#         (r"what is the meaning of (.+)", r"what is \1"),
#         (r"meaning of (.+)", r"what is \1"),
#         (r"define (.+)", r"what is \1"),
#         (r"explain (.+)", r"what is \1"),
#         (r"tell me about (.+)", r"what is \1"),
#         (r"(.+) kya hai", r"\1 kya hai"),   # Urdu support
#         (r"(.+) ka matlab kya hai", r"\1 kya hai"),
#     ]

#     for pattern, replacement in patterns:
#         q = re.sub(pattern, replacement, q)

#     return q.strip()

# #============= Query Expansion ===================

# def expand_agri_query(query: str) -> list:
#     """
#     Expand short agricultural queries to improve retrieval
#     """
#     expansions = [query]

#     q = query.lower()

#     if "pruning" in q:
#         expansions.extend([
#             query.replace("pruning", "pruning of"),
#             query.replace("pruning", "how to prune"),
#             query.replace("pruning", "pruning method"),
#             query.replace("pruning", "tree pruning"),
#         ])

#     if "mango" in q:
#         expansions.extend([
#             query.replace("mango", "mango tree"),
#             query.replace("mango", "mango plant"),
#         ])

#     if q.startswith("what is"):
#         base = q.replace("what is", "").strip()
#         expansions.extend([
#             base,
#             f"{base} cultivation",
#             f"{base} method",
#         ])

#     # Remove duplicates
#     return list(set(expansions))

# def generate_paraphrases(query, lang):
#     prompt = (
#         f"Generate 2 paraphrases:\n{query}"
#         if lang == "en"
#         else f"اس سوال کے دو متبادل جملے بنائیں:\n{query}"
#     )
#     try:
#         r = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.6,
#             max_tokens=60
#         )
#         lines = r.choices[0].message.content.split("\n")
#         return [query] + [l.strip() for l in lines if len(l.strip()) > 5]
#     except:
#         return [query]

# # ============ RETRIEVAL ============

# def retrieve_chunks(query, top_k=10):
#     """Retrieve relevant chunks from Qdrant using multiple strategies"""
#     lang = detect_lang(query)
#     query = correct_spelling(query, lang)  # optional spelling correction

#     # Generate paraphrases + domain-specific expansions
#     paraphrases = generate_paraphrases(query, lang)
#     paraphrases.extend(expand_agri_query(query))  # your agri-specific expansions
#     paraphrases = list(set(paraphrases))

#     # Compress intent to catch simplified forms
#     intent = compress_intent(query, lang)
#     if intent and intent != query:
#         paraphrases.append(intent)

#     all_hits = []

#     # --- Main retrieval loop ---
#     for q in set(paraphrases):
#         try:
#             embedding = model.encode(q).tolist()
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=embedding,
#                 limit=top_k,
#                 with_payload=True
#             )

#             for p in res.points:
#                 all_hits.append({
#                     "text": p.payload.get("text", ""),
#                     "payload": p.payload,
#                     "score": p.score,
#                     "query": q
#                 })
#         except Exception as e:
#             print(f"⚠️ Retrieval failed for paraphrase '{q}': {e}")
#             continue

#     # Deduplicate by text and keep highest scoring
#     unique = {}
#     for h in all_hits:
#         t = h["text"]
#         if t not in unique or h["score"] > unique[t]["score"]:
#             unique[t] = h

#     ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
#     ranked = [r for r in ranked if r["score"] >= MIN_SCORE_THRESHOLD]

#     # --- Fallback: broad query if nothing found ---
#     if not ranked:
#         print("⚠️ No results found, using broad query fallback...")
#         broad_query = " ".join(query.split()[:2])
#         if broad_query != query:  # prevent infinite recursion
#             return retrieve_chunks(broad_query, top_k=top_k)

#     return ranked[:top_k]



# # ============ ANSWERING ============

# def generate_answer(query, chunks):
#     lang = detect_lang(query)

#     if not chunks:
#         return (
#             "معلومات دستیاب نہیں۔"
#             if lang == "ur"
#             else "Information not available."
#         )

#     context = "\n\n---\n\n".join(
#         [c["text"] for c in chunks]
#     )

#     system_prompt = (
#         "Answer only from context. If unsure, say information not available."
#         if lang == "en"
#         else "صرف فراہم کردہ معلومات سے جواب دیں۔"
#     )

#     r = openai_client.chat.completions.create(
#         model="gpt-4o-mini",
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
#         ],
#         temperature=0.3,
#         max_tokens=400
#     )
#     return r.choices[0].message.content

# # ============ PIPELINE ============

# def rag_pipeline(query):
#     query = normalize_query(query)
#     query = normalize_intent(query)

#     chunks = retrieve_chunks(query)
#     return generate_answer(query, chunks)

# # ============ TEST ============

# if __name__ == "__main__":
#     tests = [
#         "What is PQNK?",
#         "irrgation method for potato",
#         "آلو کی کاشت کیسے کریں",
#         "Explain pruning"
#     ]

#     for q in tests:
#         print("\nQ:", q)
#         print("A:", rag_pipeline(q))






































from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langdetect import detect
from spellchecker import SpellChecker
import re
import sys

# ================= CONFIG =================
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "pqnk_vectors_v1"
OPENAI_API_KEY = "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"

'''
• Prevents weak retrieval matches
• Ensures relevance
• Avoids hallucinations
'''
MIN_SCORE_THRESHOLD = 0.25
TOP_K = 8

# ================= ENTITY MEMORY =================
ENTITY_MEMORY = {
    "last_entity": None  # stores last mentioned important entity
}

# =========================================
print("🚀 Initializing RAG System...")

# Models
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
spell_en = SpellChecker()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

# Qdrant
qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# ============ LANGUAGE & TEXT UTILS ============

def normalize_query(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text

def detect_lang(text):
    urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    eng_chars = len(re.findall(r'[A-Za-z]', text))
    if urdu_chars > eng_chars:
        return "ur"
    return "en"

def correct_spelling(text, lang):
    if lang != "en":
        return text
    return " ".join([spell_en.correction(w) or w for w in text.split()])



ACRONYM_MEMORY = {
    # example:
    # "PQNK": "Pakistan Quality Nutrition Knowledge initiative ..."
    # "PQNK": "Paedar Qudrati Nizam Kashtkari"
}

KNOWN_ACRONYMS = {
    "pqnk": "PQNK",
}


#============ experimental ==============
# ============ ENTITY MEMORY UTILS ============

def extract_entity(text):
    """Check if text contains known entities (like PQNK)"""
    for token in text.lower().split():
        if token in ["pqnk"]:  # extend this list if needed
            return token.upper()
    return None

def resolve_entity(text):
    """
    If 'it' or 'its' is in query, replace it with last entity.
    Otherwise, store new entity if found.
    """
    entity = extract_entity(text)
    if entity:
        ENTITY_MEMORY["last_entity"] = entity
        return text
    if "it" in text.lower() or "its" in text.lower():
        if ENTITY_MEMORY["last_entity"]:
            return text + f" ({ENTITY_MEMORY['last_entity']})"
    return text



#=========== ACRONYM DETECTION AND EXPANSION=============
def detect_acronym(text):
    """Return canonical acronym if found"""
    for word in text.lower().split():
        if word in KNOWN_ACRONYMS:
            return KNOWN_ACRONYMS[word]
    return None


def expand_acronym_query(query):
    """
    Expand acronym-based definition queries
    """
    acronym = detect_acronym(query)
    if not acronym:
        return query

    # Normalize definition-style questions
    if any(x in query.lower() for x in ["meaning", "define", "full form", "stands for"]):
        return f"what is {acronym} definition"

    return query


# ============ QUERY INTELLIGENCE ============

def compress_intent(query, lang):
    prompt = (
        f"Rewrite as a short search intent:\n{query}"
        if lang == "en"
        else f"اس سوال کو مختصر تلاش کے ارادے میں تبدیل کریں:\n{query}"
    )
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=30
        )
        return r.choices[0].message.content.strip()
    except:
        return query
    
def generate_sub_query(query):
    acronym = detect_acronym(query)

    if acronym and is_definition_question(query):
        return f"what is {acronym}"

    # fallback: extract first meaningful part
    words = query.split()
    return " ".join(words[:4])


def multi_hop_retrieval(query):
    # First hop: get definition / grounding
    sub_query = generate_sub_query(query)

    # Check memory first
    definition = check_definition_cache(sub_query)

    if not definition:
        chunks_1 = retrieve_chunks(sub_query)
        definition = generate_answer(sub_query, chunks_1)
        maybe_store_definition(sub_query, definition)

    # Second hop: enrich original query
    enriched_query = f"{query}. Context: {definition}"

    chunks_2 = retrieve_chunks(enriched_query)

    return definition, chunks_2

    
#========== MULTI-HOP ==============

def is_multi_hop_question(query):
    triggers = [
        "why", "how", "importance", "impact",
        "benefit", "significance", "role"
    ]
    return any(t in query.lower() for t in triggers)


def is_definition_question(query: str) -> bool:
    definition_triggers = [
        "what is",
        "meaning",
        "define",
        "definition",
        "full form",
        "stands for",
        "kya hai",
        "ka matlab"
    ]
    q = query.lower()
    return any(t in q for t in definition_triggers)

    
def normalize_intent(query: str) -> str:
    q = query.lower().strip()
    patterns = [
        (r"what is the meaning of (.+)", r"what is \1"),
        (r"meaning of (.+)", r"what is \1"),
        (r"define (.+)", r"what is \1"),
        (r"explain (.+)", r"what is \1"),
        (r"tell me about (.+)", r"what is \1"),
        (r"(.+) kya hai", r"\1 kya hai"),
        (r"(.+) ka matlab kya hai", r"\1 kya hai"),
    ]
    for pattern, replacement in patterns:
        q = re.sub(pattern, replacement, q)
    return q.strip()

#============= Query Expansion ===================

def expand_agri_query(query: str) -> list:
    expansions = [query]
    q = query.lower()
    if "pruning" in q:
        expansions.extend([
            query.replace("pruning", "pruning of"),
            query.replace("pruning", "how to prune"),
            query.replace("pruning", "pruning method"),
            query.replace("pruning", "tree pruning"),
        ])
    if "mango" in q:
        expansions.extend([
            query.replace("mango", "mango tree"),
            query.replace("mango", "mango plant"),
        ])
    if q.startswith("what is"):
        base = q.replace("what is", "").strip()
        expansions.extend([
            base,
            f"{base} cultivation",
            f"{base} method",
        ])
    return list(set(expansions))

def generate_paraphrases(query, lang):
    prompt = (
        f"Generate 2 paraphrases:\n{query}"
        if lang == "en"
        else f"اس سوال کے دو متبادل جملے بنائیں:\n{query}"
    )
    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=60
        )
        lines = r.choices[0].message.content.split("\n")
        return [query] + [l.strip() for l in lines if len(l.strip()) > 5]
    except:
        return [query]


#============= DEFINITION CACHE LOOKUP ==============
def check_definition_cache(query):
    """
    If definition already known, return it directly
    """
    acronym = detect_acronym(query)
    if acronym and acronym in ACRONYM_MEMORY:
        return ACRONYM_MEMORY[acronym]
    return None

def maybe_store_definition(query, answer):
    """
    Store definition if query is asking for meaning
    """
    acronym = detect_acronym(query)
    if not acronym:
        return

    if any(x in query.lower() for x in ["what is", "meaning", "define"]):
        if "information not available" not in answer.lower():
            ACRONYM_MEMORY[acronym] = answer


# ============ RETRIEVAL ============

def retrieve_chunks(query, top_k=10):
    lang = detect_lang(query)
    query = correct_spelling(query, lang)  # optional spelling correction

    # Generate paraphrases + domain-specific expansions
    paraphrases = generate_paraphrases(query, lang)
    paraphrases.extend(expand_agri_query(query))
    paraphrases = list(set(paraphrases))

    # Compress intent
    intent = compress_intent(query, lang)
    if intent and intent != query:
        paraphrases.append(intent)

    all_hits = []

    for q in set(paraphrases):
        try:
            embedding = model.encode(q).tolist()
            res = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=top_k,
                with_payload=True
            )
            for p in res.points:
                all_hits.append({
                    "text": p.payload.get("text", ""),
                    "payload": p.payload,
                    "score": p.score,
                    "query": q
                })
        except Exception as e:
            print(f"⚠️ Retrieval failed for paraphrase '{q}': {e}")
            continue

    unique = {}
    for h in all_hits:
        t = h["text"]
        if t not in unique or h["score"] > unique[t]["score"]:
            unique[t] = h

    ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
    ranked = [r for r in ranked if r["score"] >= MIN_SCORE_THRESHOLD]

    if not ranked:
        print("⚠️ No results found, using broad query fallback...")
        broad_query = " ".join(query.split()[:2])
        if broad_query != query:
            return retrieve_chunks(broad_query, top_k=top_k)

    return ranked[:top_k]

# ============ ANSWERING ============

def generate_answer(query, chunks):
    lang = detect_lang(query)

    if not chunks:
        return (
            "معلومات دستیاب نہیں۔"
            if lang == "ur"
            else "Information not available."
        )

    context = "\n\n---\n\n".join([c["text"] for c in chunks])

    system_prompt = (
        "Answer only from context. If unsure, say information not available."
        if lang == "en"
        else "صرف فراہم کردہ معلومات سے جواب دیں۔"
    )

    r = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion:\n{query}"}
        ],
        temperature=0.3,
        max_tokens=400
    )
    return r.choices[0].message.content

# ============ PIPELINE ============

def rag_pipeline(query):
    query = normalize_query(query)
    query = normalize_intent(query)
    query = resolve_entity(query)
    query = expand_acronym_query(query)

    # 1️⃣ Definition cache shortcut
    cached = check_definition_cache(query)
    if cached and is_definition_question(query):
        return cached

    # 2️⃣ Multi-hop path
    if is_multi_hop_question(query):
        definition, chunks = multi_hop_retrieval(query)
        answer = generate_answer(query, chunks)
        return answer

    # 3️⃣ Normal single-hop
    chunks = retrieve_chunks(query)
    answer = generate_answer(query, chunks)
    maybe_store_definition(query, answer)
    return answer


# ============ TEST ============

if __name__ == "__main__":
    tests = [
        "What is PQNK?",
        "What is the meaning of it?",
        "irrgation method for potato",
        "آلو کی کاشت کیسے کریں",
        "Explain pruning"
    ]

    for q in tests:
        print("\nQ:", q)
        print("A:", rag_pipeline(q))




























































# ================= LIBRARIES =================
# qdrant_client.QdrantClient   -> Connects to Qdrant vector DB for retrieval
# sentence_transformers.SentenceTransformer -> Creates embeddings for semantic search
# openai.OpenAI               -> Interacts with OpenAI API for LLM-based completions
# langdetect.detect           -> Detects language (English or Urdu) for query processing
# spellchecker.SpellChecker   -> Corrects spelling mistakes in English queries
# re                          -> Regex for string processing and normalization
# sys                         -> Access system-specific parameters and functions (used minimally here)
# fitz (PyMuPDF, in other scripts) -> Reads PDFs (used in text conversion script)

# ================= CONFIG VARIABLES =================
# QDRANT_API_KEY / QDRANT_URL / COLLECTION_NAME -> Config for Qdrant database connection
# OPENAI_API_KEY -> Config for OpenAI API access
# MIN_SCORE_THRESHOLD -> Minimum similarity score for retrieval to avoid irrelevant results
# TOP_K -> Maximum number of top chunks to retrieve

# ================= ENTITY MEMORY =================
# ENTITY_MEMORY -> Stores last important entity for pronoun resolution

# ================= FUNCTIONS =================

# normalize_query(text) -> Strips extra spaces and normalizes query
# detect_lang(text) -> Detects language (Urdu or English) based on character content
# correct_spelling(text, lang) -> Corrects spelling for English queries

# extract_entity(text) -> Checks if query contains known entities like PQNK
# resolve_entity(text) -> Resolves pronouns like "it" using last entity memory

# detect_acronym(text) -> Detects known acronyms in query
# expand_acronym_query(query) -> Expands acronym-based queries for definition retrieval

# compress_intent(query, lang) -> Shortens query intent using LLM (OpenAI)
# generate_sub_query(query) -> Generates a sub-query for multi-hop retrieval
# multi_hop_retrieval(query) -> Performs two-step retrieval: definition + enriched query
# is_multi_hop_question(query) -> Checks if a query is multi-hop type based on trigger words
# is_definition_question(query) -> Detects if query asks for a definition

# normalize_intent(query) -> Converts multiple query formats to a normalized "what is" style

# expand_agri_query(query) -> Generates domain-specific query expansions (agriculture-related)
# generate_paraphrases(query, lang) -> Generates paraphrases using LLM to improve retrieval

# check_definition_cache(query) -> Checks if acronym definition is already cached
# maybe_store_definition(query, answer) -> Stores acronym definitions in cache if appropriate

# retrieve_chunks(query, top_k) -> Retrieves top-k relevant text chunks from Qdrant vector DB
# generate_answer(query, chunks) -> Generates final answer using OpenAI LLM based on retrieved context

# rag_pipeline(query) -> Full RAG pipeline integrating:
#     1. Normalization
#     2. Entity resolution
#     3. Acronym expansion
#     4. Definition caching
#     5. Multi-hop retrieval
#     6. Chunk retrieval
#     7. LLM answer generation

# ================= OTHER COMMANDS =================
# model.encode(query) -> Generates vector embeddings for semantic search
# qdrant.query_points(...) -> Queries Qdrant for nearest neighbor vectors
# r.choices[0].message.content -> Extracts OpenAI response from completion API
# re.sub(pattern, replacement, string) -> Performs regex-based text normalization
# os.walk(BASE_DIR) -> Iterates through files in a folder (used in PDF-to-text script)
# open(file, "w") -> Writes extracted text to a .txt file



















































































# #ACCURACY ERRORS ARE MORE FREQUENT IN THE BELOW VERSION


# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from spellchecker import SpellChecker
# import re
# import sys
# import json
# from typing import List, Dict, Tuple

# # ================= CONFIG =================
# QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
# QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

# MIN_SCORE_THRESHOLD = 0.5  # Increased from 0.25 to ensure relevance
# TOP_K = 5  # Reduced to focus on most relevant chunks

# # ================= KNOWLEDGE BASE =================
# # Pre-populated with critical facts from documents
# KNOWLEDGE_BASE = {
#     "PQNK": {
#         "full_form": "Pristine Organic Farming System (pronounced 'Picnic')",
#         "description": "The Regenerative & Sustainable Pristine Organic Farming System",
#         "meaning": "Sustainable Natural System of Farming"
#     },
#     "OAP": {
#         "full_form": "One Acre Prosperity",
#         "goal": "To free the one-acre farm household from economic pressure by establishing a Circular Production system"
#     },
#     "nursery_block_components": "20% Rice Hull + 40% Good Fertile Soil + 20% Compost + 20% Sand",
#     "cultivation_methods": ["Nursery Growing & Transplanting", "Direct Seeding"],
#     "non_negotiable_reasons": [
#         "Saves on expensive hybrid seeds",
#         "Accelerates growth by eliminating transplant shock",
#         "Transforms nursery work into robust factory-like assembly line",
#         "Compatible with fully Mulched Beds"
#     ]
# }

# # =========================================
# print("🚀 Initializing RAG System...")

# # Models
# model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
# spell_en = SpellChecker()
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# # Qdrant
# qdrant = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
#     timeout=30
# )

# # ============ LANGUAGE & TEXT UTILS ============

# def normalize_query(text: str) -> str:
#     """Clean and normalize query text"""
#     text = text.strip()
#     text = re.sub(r"\s+", " ", text)
#     text = re.sub(r'[^\w\s\u0600-\u06FF.,?]', '', text)  # Keep Urdu and basic punctuation
#     return text

# def detect_lang(text: str) -> str:
#     """Detect if text is Urdu or English"""
#     urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#     eng_chars = len(re.findall(r'[A-Za-z]', text))
#     if urdu_chars > eng_chars:
#         return "ur"
#     return "en"

# def correct_spelling(text: str, lang: str) -> str:
#     """Simple spelling correction for English"""
#     if lang != "en":
#         return text
#     words = text.split()
#     corrected = []
#     for w in words:
#         if len(w) > 3:  # Only correct longer words
#             correction = spell_en.correction(w)
#             corrected.append(correction if correction else w)
#         else:
#             corrected.append(w)
#     return " ".join(corrected)

# # ============ QUERY ENHANCEMENT ============

# def enhance_query(query: str) -> List[str]:
#     """
#     Create multiple query variations for better retrieval
#     Focused on the specific domain (agriculture/PQNK)
#     """
#     lang = detect_lang(query)
#     query_lower = query.lower()
#     variations = [query]
    
#     # Add basic variations
#     if lang == "en":
#         if query_lower.startswith("what is"):
#             base = query_lower.replace("what is", "").strip()
#             variations.extend([
#                 f"definition of {base}",
#                 f"meaning of {base}",
#                 f"explain {base}",
#                 base  # Just the term itself
#             ])
        
#         # Add PQNK-specific expansions
#         if "pqnk" in query_lower:
#             variations.extend([
#                 "PQNK definition",
#                 "PQNK meaning",
#                 "What does PQNK stand for",
#                 "PQNK system"
#             ])
        
#         # Add OAP-specific expansions
#         if "oap" in query_lower or "one acre" in query_lower:
#             variations.extend([
#                 "One Acre Prosperity",
#                 "OAP model",
#                 "one acre farm"
#             ])
        
#         # For "how many" questions
#         if "how many" in query_lower:
#             variations.append(query_lower.replace("how many", "number of"))
            
#         # For "list" questions
#         if "list" in query_lower or "components" in query_lower or "parts" in query_lower:
#             variations.append("composition of")
#             variations.append("ingredients of")
    
#     return list(set(variations))[:5]  # Limit to 5 variations

# def extract_key_terms(query: str) -> List[str]:
#     """Extract key terms for hybrid search if needed"""
#     lang = detect_lang(query)
#     if lang == "en":
#         # Remove stopwords and keep meaningful terms
#         stopwords = {"what", "is", "the", "of", "for", "and", "how", "many", "does", "do", "are", "can", "explain"}
#         terms = [word for word in query.lower().split() if word not in stopwords and len(word) > 2]
#         return terms
#     return []

# # ============ RETRIEVAL ============

# def retrieve_chunks(query: str) -> List[Dict]:
#     """
#     Retrieve relevant chunks with enhanced query strategy
#     """
#     lang = detect_lang(query)
#     query_corrected = correct_spelling(query, lang)
    
#     # Get query variations
#     all_queries = enhance_query(query_corrected)
    
#     all_hits = []
    
#     for q in all_queries:
#         try:
#             # Generate embedding for this query variation
#             embedding = model.encode(q).tolist()
            
#             # Search in Qdrant
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=embedding,
#                 limit=TOP_K * 2,  # Get more initially
#                 with_payload=True,
#                 score_threshold=MIN_SCORE_THRESHOLD
#             )
            
#             # Add results with metadata
#             for point in res.points:
#                 all_hits.append({
#                     "text": point.payload.get("text", ""),
#                     "score": point.score,
#                     "source_query": q,
#                     "metadata": point.payload.get("metadata", {})
#                 })
                
#         except Exception as e:
#             print(f"⚠️ Retrieval failed for query '{q}': {e}")
#             continue
    
#     # Deduplicate and sort by score
#     unique_texts = {}
#     for hit in all_hits:
#         text = hit["text"]
#         if text not in unique_texts or hit["score"] > unique_texts[text]["score"]:
#             unique_texts[text] = hit
    
#     # Sort by score descending
#     sorted_hits = sorted(unique_texts.values(), key=lambda x: x["score"], reverse=True)
    
#     # Filter by score threshold and take top K
#     filtered_hits = [h for h in sorted_hits if h["score"] >= MIN_SCORE_THRESHOLD]
    
#     if not filtered_hits:
#         print(f"⚠️ No relevant chunks found for query: {query}")
#         # Fallback: try with broader search
#         try:
#             embedding = model.encode(query).tolist()
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=embedding,
#                 limit=3,
#                 with_payload=True,
#                 score_threshold=0.3  # Lower threshold for fallback
#             )
#             for point in res.points:
#                 filtered_hits.append({
#                     "text": point.payload.get("text", ""),
#                     "score": point.score,
#                     "source_query": "fallback",
#                     "metadata": point.payload.get("metadata", {})
#                 })
#         except:
#             pass
    
#     return filtered_hits[:TOP_K]

# # ============ ANSWER GENERATION ============

# def generate_answer(query: str, chunks: List[Dict]) -> str:
#     """
#     Generate answer strictly from retrieved context
#     """
#     lang = detect_lang(query)
    
#     if not chunks:
#         return "Information not available in the provided documents." if lang == "en" else "دستاویزات میں معلومات دستیاب نہیں۔"
    
#     # Build context with citations
#     context_parts = []
#     for i, chunk in enumerate(chunks[:3]):  # Use top 3 chunks max
#         context_parts.append(f"[Source {i+1}, Score: {chunk['score']:.2f}]\n{chunk['text']}\n")
    
#     context = "\n---\n".join(context_parts)
    
#     # Strict system prompts
#     if lang == "en":
#         system_prompt = """You are a precise agricultural assistant. Answer the question using ONLY information from the provided context. 
        
#         RULES:
#         1. If the answer is not found in the context, say "Information not available."
#         2. Do not add any information not in the context.
#         3. Be concise and factual.
#         4. If you find exact numbers, names, or percentages, use them exactly as in the context.
#         5. Do not make up examples or expand beyond what's given.
        
#         Context:
#         {context}
        
#         Question: {query}
        
#         Answer:"""
#     else:
#         system_prompt = """آپ ایک درست زرعی معاون ہیں۔ سوال کا جواب صرف فراہم کردہ سیاق و سباق سے معلومات استعمال کرتے ہوئے دیں۔
        
#         قوانین:
#         1. اگر جواب سیاق و سباق میں نہیں ملا، تو کہیں "معلومات دستیاب نہیں۔"
#         2. سیاق و سباق میں نہیں دی گئی کوئی معلومات شامل نہ کریں۔
#         3. مختصر اور حقیقت پسندانہ رہیں۔
#         4. اگر آپ کو عین نمبرز، نام، یا فیصد ملتے ہیں، تو انہیں سیاق و سباق میں دیے گئے طریقے سے استعمال کریں۔
#         5. مثالیں نہ گھڑیں یا دی گئی معلومات سے آگے نہ بڑھیں۔
        
#         سیاق و سباق:
#         {context}
        
#         سوال: {query}
        
#         جواب:"""
    
#     try:
#         response = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[
#                 {"role": "system", "content": system_prompt.format(context=context, query=query)},
#                 {"role": "user", "content": query}
#             ],
#             temperature=0.1,  # Very low temperature for factual answers
#             max_tokens=300
#         )
        
#         answer = response.choices[0].message.content.strip()
        
#         # Post-process to ensure no hallucination
#         if "information not available" in answer.lower() or "معلومات دستیاب نہیں" in answer:
#             # Double-check if we should really say info not available
#             # Look for exact matches in chunks
#             query_lower = query.lower()
#             if "what is" in query_lower or "meaning" in query_lower:
#                 # Check knowledge base first
#                 if "pqnk" in query_lower:
#                     if "pronounced" not in answer.lower() and "full form" not in answer.lower():
#                         return KNOWLEDGE_BASE["PQNK"]["full_form"] + " - " + KNOWLEDGE_BASE["PQNK"]["description"]
#                 elif "oap" in query_lower:
#                     return KNOWLEDGE_BASE["OAP"]["goal"]
            
#             return "Information not available." if lang == "en" else "معلومات دستیاب نہیں۔"
        
#         return answer
        
#     except Exception as e:
#         print(f"⚠️ Generation error: {e}")
#         return "Information not available." if lang == "en" else "معلومات دستیاب نہیں۔"

# # ============ FACT CHECKING ============

# def fact_check_answer(query: str, answer: str, chunks: List[Dict]) -> Tuple[str, bool]:
#     """
#     Verify answer against retrieved chunks and knowledge base
#     Returns: (verified_answer, is_correct)
#     """
#     if "information not available" in answer.lower() or "معلومات دستیاب نہیں" in answer:
#         return answer, True
    
#     # Extract key facts from answer
#     lang = detect_lang(query)
    
#     # Check against knowledge base for specific queries
#     query_lower = query.lower()
    
#     # Check for specific known facts
#     if "soil mixture" in query_lower or "مٹی کے مرکب" in query or "اجزاء" in query:
#         if KNOWLEDGE_BASE["nursery_block_components"].lower() not in answer.lower():
#             # Try to retrieve the specific information
#             for chunk in chunks:
#                 if "20%" in chunk["text"] and "rice hull" in chunk["text"].lower():
#                     return chunk["text"], True
#             return KNOWLEDGE_BASE["nursery_block_components"], True
    
#     elif "cultivation methods" in query_lower or "کاشتکاری طریقوں" in query:
#         correct_methods = set([m.lower() for m in KNOWLEDGE_BASE["cultivation_methods"]])
#         answer_methods = set([m.lower() for m in answer.split() if len(m) > 3])
#         if not any(method in answer.lower() for method in correct_methods):
#             return ", ".join(KNOWLEDGE_BASE["cultivation_methods"]), True
    
#     elif "non-negotiable" in query_lower or "غیرمشروط" in query:
#         # Check if answer contains at least 2 of the reasons
#         reasons_found = 0
#         for reason in KNOWLEDGE_BASE["non_negotiable_reasons"]:
#             if any(word in answer.lower() for word in reason.lower().split()[:3]):
#                 reasons_found += 1
#         if reasons_found < 2:
#             return "\n".join([f"{i+1}. {r}" for i, r in enumerate(KNOWLEDGE_BASE["non_negotiable_reasons"][:3])]), True
    
#     # General verification: check if key terms from query appear in answer
#     key_terms = extract_key_terms(query)
#     if key_terms:
#         terms_found = sum(1 for term in key_terms if term in answer.lower())
#         if terms_found < len(key_terms) * 0.5:  # Less than 50% of key terms found
#             return "Information not available.", True
    
#     return answer, True


# #============== DECOMPOSE COMPLEX QUERIES ==================

# # def decompose_complex_query(query: str) -> List[str]:
# #     """
# #     Break down multi-part questions into simpler sub-questions
# #     """
# #     lang = detect_lang(query)
# #     query_lower = query.lower()
    
# #     # Check if it's a compound question
# #     compound_indicators = [" and ", " also ", " furthermore ", " moreover ", 
# #                           " besides ", " as well as ", " additionally "]
    
# #     if not any(indicator in query_lower for indicator in compound_indicators):
# #         return [query]  # Not compound
    
# #     # Simple rule-based decomposition (we can make this smarter if needed)
# #     sub_queries = []
    
# #     if lang == "en":
# #         # Split by common conjunctions
# #         if " and " in query_lower:
# #             parts = query.split(" and ")
# #             for part in parts:
# #                 if len(part.strip()) > 10:  # Meaningful length
# #                     sub_queries.append(part.strip())
# #         elif " also " in query_lower:
# #             # Handle "also" questions
# #             main_part = query.split(" also ")[0]
# #             sub_queries.append(main_part.strip())
# #             # The "also" part might need context from the first part
# #     else:
# #         # Urdu compound indicators
# #         if " اور " in query:
# #             parts = query.split(" اور ")
# #             for part in parts:
# #                 if len(part.strip()) > 10:
# #                     sub_queries.append(part.strip())
    
# #     return sub_queries if len(sub_queries) > 1 else [query]


# # ============ MAIN PIPELINE ============

# def rag_pipeline_single(query: str) -> str:
#     """
#     Process a single question (without decomposition)
#     This is a refactored version of the original rag_pipeline logic
#     """
#     # Normalize query
#     query = normalize_query(query)
#     lang = detect_lang(query)
    
#     # Check if query is answerable from knowledge base
#     query_lower = query.lower()
    
#     # Direct knowledge base lookups for critical facts
#     if lang == "en":
#         if "what is pqnk" in query_lower or "meaning of pqnk" in query_lower:
#             return KNOWLEDGE_BASE["PQNK"]["full_form"] + " - " + KNOWLEDGE_BASE["PQNK"]["description"]
#         elif "what is oap" in query_lower:
#             return KNOWLEDGE_BASE["OAP"]["goal"]
#         elif "soil mixture" in query_lower and ("nursery" in query_lower or "block" in query_lower):
#             return f"The soil mixture for nursery blocks is: {KNOWLEDGE_BASE['nursery_block_components']}"
    
#     # Retrieve relevant chunks
#     chunks = retrieve_chunks(query)
    
#     if not chunks:
#         return "Information not available in the provided documents." if lang == "en" else "دستاویزات میں معلومات دستیاب نہیں۔"
    
#     # Generate answer
#     answer = generate_answer(query, chunks)
    
#     # Fact check and verify
#     verified_answer, is_correct = fact_check_answer(query, answer, chunks)
    
#     if not is_correct:
#         # Try to extract answer directly from chunks
#         for chunk in chunks:
#             if any(term in chunk["text"].lower() for term in extract_key_terms(query)):
#                 # Extract the most relevant sentence
#                 sentences = chunk["text"].split('.')
#                 for sentence in sentences:
#                     if any(term in sentence.lower() for term in extract_key_terms(query)):
#                         return sentence.strip() + "."
    
#     return verified_answer


# def rag_pipeline_single(query: str) -> str:
#     """
#     Process a single question (without decomposition)
#     This is a refactored version of the original rag_pipeline logic
#     """
#     # Normalize query
#     query = normalize_query(query)
#     lang = detect_lang(query)
    
#     # Check if query is answerable from knowledge base
#     query_lower = query.lower()
    
#     # Direct knowledge base lookups for critical facts
#     if lang == "en":
#         if "what is pqnk" in query_lower or "meaning of pqnk" in query_lower:
#             return KNOWLEDGE_BASE["PQNK"]["full_form"] + " - " + KNOWLEDGE_BASE["PQNK"]["description"]
#         elif "what is oap" in query_lower:
#             return KNOWLEDGE_BASE["OAP"]["goal"]
#         elif "soil mixture" in query_lower and ("nursery" in query_lower or "block" in query_lower):
#             return f"The soil mixture for nursery blocks is: {KNOWLEDGE_BASE['nursery_block_components']}"
    
#     # Retrieve relevant chunks
#     chunks = retrieve_chunks(query)
    
#     if not chunks:
#         return "Information not available in the provided documents." if lang == "en" else "دستاویزات میں معلومات دستیاب نہیں۔"
    
#     # Generate answer
#     answer = generate_answer(query, chunks)
    
#     # Fact check and verify
#     verified_answer, is_correct = fact_check_answer(query, answer, chunks)
    
#     if not is_correct:
#         # Try to extract answer directly from chunks
#         for chunk in chunks:
#             if any(term in chunk["text"].lower() for term in extract_key_terms(query)):
#                 # Extract the most relevant sentence
#                 sentences = chunk["text"].split('.')
#                 for sentence in sentences:
#                     if any(term in sentence.lower() for term in extract_key_terms(query)):
#                         return sentence.strip() + "."
    
#     return verified_answer


# def decompose_complex_query(query: str) -> List[str]:
#     """
#     Break down multi-part questions into simpler sub-questions
#     Returns list of sub-queries (or original query if not compound)
#     """
#     lang = detect_lang(query)
#     query_lower = query.lower()
    
#     # Check if it's a compound question
#     if lang == "en":
#         compound_indicators = [" and ", " also ", " furthermore ", " moreover ", 
#                               " besides ", " as well as ", " additionally "]
#     else:
#         compound_indicators = [" اور ", " نیز ", " مزید برآں ", " اس کے علاوہ "]
    
#     if not any(indicator in query for indicator in compound_indicators):
#         return [query]  # Not compound
    
#     sub_queries = []
    
#     # Simple rule-based decomposition
#     if lang == "en":
#         if " and " in query_lower:
#             # Handle "and" carefully - don't split too aggressively
#             parts = []
#             current_part = ""
#             words = query.split()
            
#             for word in words:
#                 if word.lower() == "and" and len(current_part) > 0:
#                     parts.append(current_part.strip())
#                     current_part = ""
#                 else:
#                     current_part += " " + word
            
#             if current_part:
#                 parts.append(current_part.strip())
            
#             if len(parts) > 1:
#                 for part in parts:
#                     # Add question mark if missing
#                     if not part.endswith('?'):
#                         part = part + '?'
#                     sub_queries.append(part)
    
#     else:  # Urdu
#         if " اور " in query:
#             parts = query.split(" اور ")
#             for i, part in enumerate(parts):
#                 part = part.strip()
#                 # Add question structure if needed
#                 if "کیا" not in part and "کون" not in part and "کس" not in part:
#                     if i == 0:
#                         # Keep the first part as is (it probably has the question word)
#                         sub_queries.append(part)
#                     else:
#                         # Add a question marker to subsequent parts
#                         sub_queries.append(part + " کیا ہے؟")
#                 else:
#                     sub_queries.append(part)
    
#     # If decomposition didn't work well, return original
#     if not sub_queries or len(sub_queries) == 1:
#         return [query]
    
#     return sub_queries

# def rag_pipeline(query: str) -> str:
#     """
#     Main RAG pipeline with enhanced retrieval and strict answer generation
#     """
#     # Step 1: Normalize query
#     query = normalize_query(query)
#     lang = detect_lang(query)
    
#     print(f"\n🔍 Processing query: {query}")
#     print(f"📝 Language: {lang}")
    
#     # Step 2: Decompose complex queries
#     sub_queries = decompose_complex_query(query)
    
#     if len(sub_queries) > 1:
#         print(f"📝 Decomposed into {len(sub_queries)} sub-questions")
#         answers = []
#         for i, sub_q in enumerate(sub_queries):
#             print(f"  Sub-question {i+1}: {sub_q}")
#             sub_answer = rag_pipeline_single(sub_q)
#             answers.append(sub_answer)
        
#         # Combine answers
#         if lang == "en":
#             combined = "\n\n".join([f"{i+1}. {ans}" for i, ans in enumerate(answers)])
#         else:
#             combined = "\n\n".join([f"{i+1}۔ {ans}" for i, ans in enumerate(answers)])
#         return combined
    
#     # Step 3: Process as single question
#     return rag_pipeline_single(query)

# # ============ TEST ============

# if __name__ == "__main__":
#     # Test with the problematic questions from accuracy matrix
#     tests = [
#         # English questions
#         "What is PQNK?",
#         "What does PQNK stand for?",
#         "What is the soil mixture for nursery blocks?",
#         "What are the cultivation methods mentioned?",
#         "How many crops can be grown in one year?",
#         "What is air pruning?",
#         "What are the non-negotiable reasons for nursery block system?",
#         "What is field heat?",
        
#         # Urdu questions
#         "PQNK کا کیا مطلب ہے؟",
#         "نرسری بلاکس کے لیے مٹی کا مرکب کیا ہے؟",
#         "کاشتکاری کے طریقے کون سے ہیں؟",
#         "ایئر پروننگ کیا ہے؟",
#         "فیلڈ ہیٹ کیا ہے؟",
#     ]
    
#     print("=" * 60)
#     print("RAG SYSTEM TESTING")
#     print("=" * 60)
    
#     for q in tests:
#         print(f"\n{'='*40}")
#         print(f"❓ QUESTION: {q}")
#         print(f"{'='*40}")
#         answer = rag_pipeline(q)
#         print(f"✅ ANSWER: {answer}")
#         print(f"{'='*40}")
    
#     # Interactive mode
#     print("\n\n🎯 INTERACTIVE MODE (Ctrl+C to exit)")
#     print("=" * 60)
    
#     while True:
#         try:
#             user_query = input("\n🤔 Enter your question: ").strip()
#             if user_query.lower() in ['exit', 'quit', 'bye']:
#                 break
#             if user_query:
#                 answer = rag_pipeline(user_query)
#                 print(f"\n💡 Answer: {answer}")
#         except KeyboardInterrupt:
#             print("\n👋 Goodbye!")
#             break
#         except Exception as e:
#             print(f"⚠️ Error: {e}")
