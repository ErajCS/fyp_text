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













































# import psycopg2
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from pgvector.psycopg2 import register_vector
# import os

# # --- CONFIG ---
# DB_NAME = "pqnk_db"
# DB_USER = "postgres"
# DB_PASS = "admin123"  # <--- CONFIRM YOUR PASSWORD
# # Ensure your API Key is set here or in environment variables
#
# # Initialize Models
# print("Loading models...")
# model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# try:
#     client = OpenAI(api_key=OPENAI_API_KEY)
# except:
#     print("⚠️ OpenAI API Key missing.")
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
#     """
#     Retrieves top k chunks using HYBRID SEARCH:
#     Combines Vector Similarity (Semantic) + Full-Text Search (Keyword Match).
#     """
#     query_vec = model.encode(query).tolist()
#     conn = get_db_connection()
#     cur = conn.cursor()
    
#     # 1. Prepare Keyword Search Query
#     # Convert query to tsquery format. 'plainto_tsquery' parses natural language 
#     # into valid search tokens (e.g. "soil strategy" -> "'soil' & 'strategi'")
#     ts_query = query.replace("'", "''") 
    
#     # 2. HYBRID QUERY LOGIC
#     # Formula: Hybrid Score = (Vector Distance) - (Keyword Rank * Weight)
#     # - Vector Distance (<=>): Lower is better (0=identical, 1=different)
#     # - Keyword Rank (ts_rank): Higher is better (0=no match, 0.1+=match)
#     #
#     # We subtract the Keyword Rank from the Distance. 
#     # Since we ORDER BY score ASC (lowest first), subtracting a high keyword score 
#     # pushes that result to the top (making the score smaller/negative).
#     # Weight (0.2) controls how much keywords matter vs meaning.
    
#     sql = """
#         SELECT 
#             v.chunk_text, 
#             c.filename, 
#             (v.embedding <=> %s::vector) - (ts_rank(v.text_search_tokens, plainto_tsquery('english', %s)) * 0.2) as hybrid_score
#         FROM vectors v
#         JOIN content c ON v.content_id = c.content_id
#         WHERE c.language = %s
#         ORDER BY hybrid_score ASC
#         LIMIT %s
#     """
    
#     # Execute query with parameters
#     cur.execute(sql, (query_vec, ts_query, lang, top_k))
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
    
#     # 1. Retrieve Passages (Hybrid Search)
#     passages = retrieve_passages_from_db(question, lang)
    
#     if not passages:
#         return "⚠️ No relevant information found in the PQNK knowledge base."
        
#     # 2. Generate Answer (from GPT)
#     answer = generate_gpt_response(question, passages)
#     return answer

# if __name__ == "__main__":
#     # Test locally
#     print("🚀 Database-Powered HYBRID RAG Testing Mode")
#     while True:
#         q = input("Ask (or exit): ")
#         if q == "exit": break
#         print("\n🤖 Answer:\n" + rag_pipeline(q))













































# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# import numpy as np
# import re
# import qdrant_client  # To check version

# # ================= CONFIG =================
#AI_
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
# QDRAN
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





















































# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from spellchecker import SpellChecker
# import re
# import sys

# # ================= CONFIG =================
# QDRANT
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








































#========================== WORKING CODE ===============================================


# from qdrant_client import QdrantClient
# from sentence_transformers import SentenceTransformer
# from openai import OpenAI
# from langdetect import detect
# from spellchecker import SpellChecker
# import re
# import sys
# import os
# import time
# import socket
# from collections import defaultdict

# # ================= CONFIG =================
# # QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# # OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
# # QDRANT_URL = os.getenv("QDRANT_URL")
# # COLLECTION_NAME = "pqnk_vectors_v1"
# ME = "pqnk_v2"

# '''
# • Prevents weak retrieval matches
# • Ensures relevance
# • Avoids hallucinations
# '''
# MIN_SCORE_THRESHOLD = 0.25
# TOP_K = 8

# # ================= ENTITY MEMORY =================
# ENTITY_MEMORY = {
#     "last_entity": None  # stores last mentioned important entity
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



# ACRONYM_MEMORY = {
#     # example:
#     # "PQNK": "Pakistan Quality Nutrition Knowledge initiative ..."
#     # "PQNK": "Paedar Qudrati Nizam Kashtkari"
# }

# KNOWN_ACRONYMS = {
#     "pqnk": "PQNK",
# }


# #============ experimental ==============
# # ============ ENTITY MEMORY UTILS ============

# def extract_entity(text):
#     """Check if text contains known entities (like PQNK)"""
#     for token in text.lower().split():
#         if token in ["pqnk"]:  # extend this list if needed
#             return token.upper()
#     return None

# def resolve_entity(text):
#     """
#     If 'it' or 'its' is in query, replace it with last entity.
#     Otherwise, store new entity if found.
#     """
#     entity = extract_entity(text)
#     if entity:
#         ENTITY_MEMORY["last_entity"] = entity
#         return text
#     if "it" in text.lower() or "its" in text.lower():
#         if ENTITY_MEMORY["last_entity"]:
#             return text + f" ({ENTITY_MEMORY['last_entity']})"
#     return text



# #=========== ACRONYM DETECTION AND EXPANSION=============
# def detect_acronym(text):
#     """Return canonical acronym if found"""
#     for word in text.lower().split():
#         if word in KNOWN_ACRONYMS:
#             return KNOWN_ACRONYMS[word]
#     return None


# def expand_acronym_query(query):
#     """
#     Expand acronym-based definition queries
#     """
#     acronym = detect_acronym(query)
#     if not acronym:
#         return query

#     # Normalize definition-style questions
#     if any(x in query.lower() for x in ["meaning", "define", "full form", "stands for"]):
#         return f"what is {acronym} definition"

#     return query


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
    
# def generate_sub_query(query):
#     acronym = detect_acronym(query)

#     if acronym and is_definition_question(query):
#         return f"what is {acronym}"

#     # fallback: extract first meaningful part
#     words = query.split()
#     return " ".join(words[:4])


# def multi_hop_retrieval(query):
#     # First hop: get definition / grounding
#     sub_query = generate_sub_query(query)

#     # Check memory first
#     definition = check_definition_cache(sub_query)

#     if not definition:
#         chunks_1 = retrieve_chunks(sub_query)
#         definition = generate_answer(sub_query, chunks_1)
#         maybe_store_definition(sub_query, definition)

#     # Second hop: enrich original query
#     enriched_query = f"{query}. Context: {definition}"

#     chunks_2 = retrieve_chunks(enriched_query)

#     return definition, chunks_2

    
# #========== MULTI-HOP ==============

# def is_multi_hop_question(query):
#     triggers = [
#         "why", "how", "importance", "impact",
#         "benefit", "significance", "role"
#     ]
#     return any(t in query.lower() for t in triggers)


# def is_definition_question(query: str) -> bool:
#     definition_triggers = [
#         "what is",
#         "meaning",
#         "define",
#         "definition",
#         "full form",
#         "stands for",
#         "kya hai",
#         "ka matlab"
#     ]
#     q = query.lower()
#     return any(t in q for t in definition_triggers)

    
# def normalize_intent(query: str) -> str:
#     q = query.lower().strip()
#     patterns = [
#         (r"what is the meaning of (.+)", r"what is \1"),
#         (r"meaning of (.+)", r"what is \1"),
#         (r"define (.+)", r"what is \1"),
#         (r"explain (.+)", r"what is \1"),
#         (r"tell me about (.+)", r"what is \1"),
#         (r"(.+) kya hai", r"\1 kya hai"),
#         (r"(.+) ka matlab kya hai", r"\1 kya hai"),
#     ]
#     for pattern, replacement in patterns:
#         q = re.sub(pattern, replacement, q)
#     return q.strip()

# #============= Query Expansion ===================

# def expand_agri_query(query: str) -> list:
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


# #============= DEFINITION CACHE LOOKUP ==============
# def check_definition_cache(query):
#     """
#     If definition already known, return it directly
#     """
#     acronym = detect_acronym(query)
#     if acronym and acronym in ACRONYM_MEMORY:
#         return ACRONYM_MEMORY[acronym]
#     return None

# def maybe_store_definition(query, answer):
#     """
#     Store definition if query is asking for meaning
#     """
#     acronym = detect_acronym(query)
#     if not acronym:
#         return

#     if any(x in query.lower() for x in ["what is", "meaning", "define"]):
#         if "information not available" not in answer.lower():
#             ACRONYM_MEMORY[acronym] = answer


# # ============ RETRIEVAL ============

# def retrieve_chunks(query, top_k=8):
#     lang = detect_lang(query)
#     query = correct_spelling(query, lang)  

#     # Generate paraphrases + domain-specific expansions
#     paraphrases = generate_paraphrases(query, lang)
#     paraphrases.extend(expand_agri_query(query))
#     paraphrases = list(set(paraphrases))

#     # Compress intent
#     intent = compress_intent(query, lang)
#     if intent and intent != query:
#         paraphrases.append(intent)

#     all_hits = []

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

#     unique = {}
#     for h in all_hits:
#         t = h["text"]
#         if t not in unique or h["score"] > unique[t]["score"]:
#             unique[t] = h

#     ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
#     ranked = [r for r in ranked if r["score"] >= MIN_SCORE_THRESHOLD]

#     if not ranked:
#         print("⚠️ No results found, using broad query fallback...")
#         broad_query = " ".join(query.split()[:2])
#         if broad_query != query:
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

#     context = "\n\n---\n\n".join([c["text"] for c in chunks])

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
#     query = resolve_entity(query)
#     query = expand_acronym_query(query)

#     # 1️⃣ Definition cache shortcut
#     cached = check_definition_cache(query)
#     if cached and is_definition_question(query):
#         return cached

#     # 2️⃣ Multi-hop path
#     if is_multi_hop_question(query):
#         definition, chunks = multi_hop_retrieval(query)
#         answer = generate_answer(query, chunks)
#         return answer

#     # 3️⃣ Normal single-hop
#     chunks = retrieve_chunks(query)
#     answer = generate_answer(query, chunks)
#     maybe_store_definition(query, answer)
#     return answer


# # ============ TEST ============

# if __name__ == "__main__":
#     tests = [
#         "What is PQNK?",
#         "What is the meaning of it?",
#         "irrgation method for potato",
#         "آلو کی کاشت کیسے کریں",
#         "Explain pruning"
#     ]

#     for q in tests:
#         print("\nQ:", q)
#         print("A:", rag_pipeline(q))



#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX













# import os
# import sys

# # 1. Enable Windows Color Support (Crucial for Windows users)
# os.system("")

# from qdrant_client import QdrantClient
# from openai import OpenAI
# from spellchecker import SpellChecker
# import re
# import time

# from rich.console import Console  # <--- NEW
# from rich.markdown import Markdown # <--- NEW
# from dotenv import load_dotenv
# load_dotenv()

# # import os

# # ================= CONFIG =================
# # Force terminal to recognize colors even in restricted environments
# console = Console(force_terminal=True)

# # 🔑 API KEYS
# # import os

# QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
# OPENAI_API_KEY = "sk-prLnIlLrGvTXV-t1z0KWHu91YHBXJwBQnqJQnvYA"
# QDRANT_URL = os.getenv("QDRANT_URL")

# COLLECTION_NAME = "pqnk_v2" 

# # ⚙️ MODEL SETTINGS
# GENERATION_MODEL = "gpt-4o"  
# EMBEDDING_MODEL = "text-embedding-3-small" # <--- Must match ingest_data.py

# # 🛡️ RETRIEVAL SAFEGUARDS
# MIN_SCORE_THRESHOLD = 0.25
# TOP_K = 8

# # ================= ENTITY MEMORY =================
# ENTITY_MEMORY = { "last_entity": None }
# ACRONYM_MEMORY = {}
# KNOWN_ACRONYMS = { "pqnk": "PQNK" }

# # =========================================
# print(f"🚀 Initializing RAG System linked to {COLLECTION_NAME}...")

# # Models
# # NOTE: SentenceTransformer is REMOVED. We use OpenAI for everything now.
# spell_en = SpellChecker()
# openai_client = OpenAI(api_key=OPENAI_API_KEY)

# # Qdrant
# qdrant = QdrantClient(
#     url=QDRANT_URL,
#     api_key=QDRANT_API_KEY,
#     timeout=30
# )

# # ============ CORE UTILS ============

# def get_embedding(text):
#     """
#     Generates vector using OpenAI to match the stored data.
#     """
#     text = text.replace("\n", " ")
#     return openai_client.embeddings.create(
#         input=[text], 
#         model=EMBEDDING_MODEL
#     ).data[0].embedding

# def normalize_query(text):
#     text = text.strip()
#     text = re.sub(r"\s+", " ", text)
#     return text

# def detect_lang(text):
#     try:
#         urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
#         eng_chars = len(re.findall(r'[A-Za-z]', text))
#         if urdu_chars > eng_chars: return "ur"
#         return "en"
#     except: return "en"

# def correct_spelling(text, lang):
#     if lang != "en": return text
#     return " ".join([spell_en.correction(w) or w if w.isalpha() else w for w in text.split()])

# # ============ ENTITY & ACRONYM LOGIC ============

# def extract_entity(text):
#     tokens = re.findall(r'\b\w+\b', text.lower())
#     for token in tokens:
#         if token in ["pqnk"]: return token.upper()
#     return None

# def resolve_entity(text):
#     entity = extract_entity(text)
#     if entity:
#         ENTITY_MEMORY["last_entity"] = entity
#         return text
#     if "it" in text.lower() or "its" in text.lower():
#         if ENTITY_MEMORY["last_entity"]:
#             return text + f" ({ENTITY_MEMORY['last_entity']})"
#     return text

# def detect_acronym(text):
#     for word in text.lower().split():
#         clean = re.sub(r'\W+', '', word)
#         if clean in KNOWN_ACRONYMS: return KNOWN_ACRONYMS[clean]
#     return None

# def expand_acronym_query(query):
#     acronym = detect_acronym(query)
#     if not acronym: return query
#     if any(x in query.lower() for x in ["meaning", "define", "full form"]):
#         return f"what is {acronym} definition"
#     return query

# # ============ INTELLIGENCE LAYER ============

# def compress_intent(query, lang):
#     prompt = (f"Rewrite as a short search intent:\n{query}" if lang == "en" 
#               else f"اس سوال کو مختصر تلاش کے ارادے میں تبدیل کریں:\n{query}")
#     try:
#         r = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0, max_tokens=30
#         )
#         return r.choices[0].message.content.strip()
#     except: return query

# def is_multi_hop_question(query):
#     triggers = ["why", "how", "importance", "impact", "benefit", "significance", "role"]
#     return any(t in query.lower() for t in triggers)

# def is_definition_question(query):
#     triggers = ["what is", "meaning", "define", "definition", "kya hai"]
#     return any(t in query.lower() for t in triggers)

# def normalize_intent(query):
#     q = query.lower().strip()
#     patterns = [
#         (r"what is the meaning of (.+)", r"what is \1"),
#         (r"define (.+)", r"what is \1"),
#     ]
#     for p, r in patterns: q = re.sub(p, r, q)
#     return q.strip()

# def expand_agri_query(query):
#     expansions = [query]
#     q = query.lower()
#     if "pruning" in q:
#         expansions.extend([query.replace("pruning", "pruning method"), query.replace("pruning", "how to prune")])
#     if "mango" in q:
#         expansions.extend([query.replace("mango", "mango tree")])
#     return list(set(expansions))

# def generate_paraphrases(query, lang):
#     prompt = (f"Generate 2 paraphrases:\n{query}" if lang == "en" else f"اس سوال کے دو متبادل جملے بنائیں:\n{query}")
#     try:
#         r = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.6, max_tokens=60
#         )
#         lines = r.choices[0].message.content.split("\n")
#         return [query] + [l.strip() for l in lines if len(l.strip()) > 5]
#     except: return [query]

# # ============ RETRIEVAL (WITH TERMINAL DEBUGGING) ============

# def retrieve_chunks(query, top_k=8):
#     lang = detect_lang(query)
#     query = correct_spelling(query, lang)  

#     # Expand Query
#     paraphrases = generate_paraphrases(query, lang)
#     paraphrases.extend(expand_agri_query(query))
#     intent = compress_intent(query, lang)
#     if intent and intent != query: paraphrases.append(intent)
    
#     unique_queries = list(set(paraphrases))

#     print(f"\n🔎 Processing {len(unique_queries)} variations for: '{query}'")
#     print("-" * 60)

#     all_hits = []

#     for q in unique_queries:
#         try:
#             # 🟢 UPDATED: Using OpenAI Embedding instead of local model
#             embedding = get_embedding(q)
            
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=embedding,
#                 limit=top_k,
#                 with_payload=True
#             )
            
#             # --- DEBUG PRINT ---
#             if res.points:
#                 print(f"   Query: '{q}'")
#                 for p in res.points:
#                     doc = p.payload.get("doc_name", "Unknown")
#                     cat = p.payload.get("category", "N/A")
#                     print(f"     • [Score: {p.score:.4f}] {doc} ({cat})")
#                     all_hits.append({
#                         "text": p.payload.get("text", ""),
#                         "payload": p.payload,
#                         "score": p.score
#                     })
#         except Exception as e:
#             print(f"⚠️ Retrieval failed for '{q}': {e}")
#             continue

#     print("-" * 60)

#     # Deduplication
#     unique = {}
#     for h in all_hits:
#         t = h["text"]
#         if t not in unique or h["score"] > unique[t]["score"]:
#             unique[t] = h

#     ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)
#     final_chunks = [r for r in ranked if r["score"] >= MIN_SCORE_THRESHOLD][:top_k]

#     # --- FINAL DEBUG PRINT ---
#     print(f"✅ Final Top-{len(final_chunks)} Chunks Passed to GPT-4o:")
#     if not final_chunks:
#         print("   ❌ No chunks met the threshold.")
#     else:
#         for i, chunk in enumerate(final_chunks):
#             doc = chunk["payload"].get("doc_name", "Unknown")
#             print(f"   {i+1}. [Score: {chunk['score']:.4f}] {doc}")
#             print(f"      Preview: \"{chunk['text'][:80].replace(chr(10), ' ')}...\"")
#     print("=" * 60 + "\n")

#     return final_chunks

# def multi_hop_retrieval(query):
#     print("\n🐰 Hop 1: Definition Search")
#     sub_query = generate_sub_query(query) # Using helper function
    
#     chunks_1 = retrieve_chunks(sub_query)
#     definition = generate_answer(sub_query, chunks_1)
    
#     # Store context
#     grounding_text = definition[:500] 

#     print(f"\n🐰 Hop 2: Enriched Context Search")
#     enriched_query = f"{query}. Context: {grounding_text}"
#     chunks_2 = retrieve_chunks(enriched_query)

#     # Combine chunks from both hops
#     combined_map = {hash(c["text"]): c for c in chunks_1 + chunks_2}
#     return list(combined_map.values())

# def generate_sub_query(query):
#     # Quick helper for multi-hop
#     words = query.split()
#     return " ".join(words[:4])

# # ============ ANSWERING ============

# # ============ ANSWERING ============

# def generate_answer(original_query, chunks, target_lang="en"):
#     # 1. Safety Check
#     if not chunks:
#         return ("معلومات دستیاب نہیں۔" if target_lang == "ur" else "Information not available.")

#     # 2. Prepare Context
#     context_text = ""
#     for c in chunks:
#         doc = c['payload'].get('doc_name', 'Unknown')
#         context_text += f"Source: {doc}\nContent: {c['text']}\n\n"

#     # 3. Define Prompts with Formatting Instructions
#     if target_lang == "ur":
#         system_prompt = (
#             "You are an agricultural expert. You will receive context in English. "
#             "You must answer the user's question in clear, professional Urdu. "
#             "IMPORTANT FORMATTING RULES:\n"
#             "1. Use **Urdu Numerals** (۱, ۲, ۳) for lists followed by a dash (e.g., ۱- متن).\n"
#             "2. Do NOT use English/Standard Markdown numbering (like 1. or 1-).\n"
#             "3. Do NOT mention sources inside the paragraphs.\n"
#             "4. At the very end, leave a blank line and list sources under '### حوالہ جات:'."
#         )
#     else:
#         system_prompt = (
#             "You are a helpful assistant. Answer ONLY using the provided Context. "
#             "Do NOT cite sources inside the text sentences. "
#             "At the very end of your response, leave a blank line and list the unique source names under the heading '### Sources:'."
#         )

#     # 4. Generate
#     r = openai_client.chat.completions.create(
#         model=GENERATION_MODEL, # GPT-4o
#         messages=[
#             {"role": "system", "content": system_prompt},
#             {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{original_query}"}
#         ],
#         temperature=0.3,
#         max_tokens=800
#         # retrieve_chunks = 8
#     )
#     return r.choices[0].message.content

# # ============ PIPELINE ============

# def rag_pipeline(query):
#     print(f"\n💬 USER QUERY: {query}")
#     query = normalize_query(query)
#     query = normalize_intent(query)
#     query = resolve_entity(query)
#     query = expand_acronym_query(query)

#     if is_multi_hop_question(query):
#         chunks = multi_hop_retrieval(query) # Now returns combined chunks
#         answer = generate_answer(query, chunks)
#         return answer

#     chunks = retrieve_chunks(query)
#     answer = generate_answer(query, chunks)
#     return answer

# # ============ TEST ============

# # ============ TEST ============

# if __name__ == "__main__":
#     # Test queries
#     tests = [
#         "What is PQNK?",
#         "How to prune raddish?",
#         "آلو کی کاشت کے لیے پانی", 
#     ]

#     print("\n" + "="*50)
#     print("🤖 AGRICULTURAL RAG CHATBOT (Formatted Output)")
#     print("="*50 + "\n")

#     for q in tests:
#         # 1. Show the Question nicely
#         console.print(f"[bold cyan]USER:[/bold cyan] {q}")
        
#         # 2. Get the raw answer
#         raw_answer = rag_pipeline(q)
        
#         # 3. Render the Markdown (Bold, Headings, Lists)
#         console.print(f"[bold green]BOT:[/bold green]")
#         console.print(Markdown(raw_answer))
        
#         print("-" * 50 + "\n")




































import os
import sys
import re
import time

# 1. Enable Windows Color Support (Crucial for Windows users)
os.system("")

from qdrant_client import QdrantClient
from openai import OpenAI
from spellchecker import SpellChecker

from rich.console import Console
from rich.markdown import Markdown
from dotenv import load_dotenv

load_dotenv()

# ================= CONFIG =================

console = Console(force_terminal=True)

QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# ⚠️ API key must be in .env — never hardcode here!
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


COLLECTION_NAME = "pqnk_v2"

# ⚙️ MODEL SETTINGS
GENERATION_MODEL = "gpt-4o"
EMBEDDING_MODEL = "text-embedding-3-small"

# 🛡️ RETRIEVAL SAFEGUARDS
MIN_SCORE_THRESHOLD = 0.25
TOP_K = 8

# ================= MEMORY =================
ENTITY_MEMORY = {"last_entity": None}
ACRONYM_MEMORY = {}
KNOWN_ACRONYMS = {"pqnk": "PQNK"}

# =========================================
print(f"🚀 Initializing RAG System linked to {COLLECTION_NAME}...")

spell_en = SpellChecker()
openai_client = OpenAI(api_key=OPENAI_API_KEY)

qdrant = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=30
)

# ============ CORE UTILS ============

def get_embedding(text):
    """
    Generates embedding using OpenAI to match the stored data.
    """
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    ).data[0].embedding


def normalize_query(text):
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_lang(text):
    """
    Simple but very reliable Urdu/English detector.
    """
    try:
        urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
        eng_chars = len(re.findall(r'[A-Za-z]', text))
        if urdu_chars > eng_chars:
            return "ur"
        return "en"
    except:
        return "en"


def correct_spelling(text, lang):
    """
    Spell correction only for English.
    """
    if lang != "en":
        return text
    return " ".join([
        (spell_en.correction(w) or w) if w.isalpha() else w
        for w in text.split()
    ])


# ============ TRANSLATION (NEW) ============

def translate_ur_to_en(urdu_text):
    """
    Translate Urdu query to English for better retrieval.
    Uses gpt-4o-mini (fast + cheap).
    """
    prompt = (
        "Translate this Urdu agricultural question into English.\n"
        "Rules:\n"
        "1) Keep technical terms accurate\n"
        "2) Keep it short like a search query\n"
        "3) Do NOT add extra information\n\n"
        f"Urdu:\n{urdu_text}"
    )

    try:
        r = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=80
        )
        return r.choices[0].message.content.strip()
    except:
        return urdu_text


# ============ ENTITY & ACRONYM LOGIC ============

def extract_entity(text):
    tokens = re.findall(r'\b\w+\b', text.lower())
    for token in tokens:
        if token in ["pqnk"]:
            return token.upper()
    return None


def resolve_entity(text):
    entity = extract_entity(text)
    if entity:
        ENTITY_MEMORY["last_entity"] = entity
        return text

    if "it" in text.lower() or "its" in text.lower():
        if ENTITY_MEMORY["last_entity"]:
            return text + f" ({ENTITY_MEMORY['last_entity']})"

    return text


def detect_acronym(text):
    for word in text.lower().split():
        clean = re.sub(r'\W+', '', word)
        if clean in KNOWN_ACRONYMS:
            return KNOWN_ACRONYMS[clean]
    return None


def expand_acronym_query(query):
    acronym = detect_acronym(query)
    if not acronym:
        return query

    if any(x in query.lower() for x in ["meaning", "define", "full form"]):
        return f"what is {acronym} definition"

    return query


# ============ INTELLIGENCE LAYER ============

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


def is_multi_hop_question(query):
    triggers = ["why", "how", "importance", "impact", "benefit", "significance", "role"]
    return any(t in query.lower() for t in triggers)


def normalize_intent(query):
    q = query.lower().strip()
    patterns = [
        (r"what is the meaning of (.+)", r"what is \1"),
        (r"define (.+)", r"what is \1"),
    ]
    for p, r in patterns:
        q = re.sub(p, r, q)
    return q.strip()


def expand_agri_query(query):
    expansions = [query]
    q = query.lower()

    if "pruning" in q:
        expansions.extend([
            query.replace("pruning", "pruning method"),
            query.replace("pruning", "how to prune")
        ])

    if "mango" in q:
        expansions.extend([query.replace("mango", "mango tree")])

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


# ============ RETRIEVAL HELPERS (NEW) ============

def language_boost_score(hit, query_lang):
    """
    Softly boost score if chunk language matches query language.
    (Not a hard filter, just improves ranking.)
    """
    payload_lang = hit["payload"].get("language", "unknown")

    if query_lang == "ur" and payload_lang == "ur":
        return hit["score"] + 0.03
    if query_lang == "en" and payload_lang == "en":
        return hit["score"] + 0.03

    return hit["score"]


def retrieve_chunks_single(query, top_k=8, debug_label=""):
    """
    Original retrieval logic (single query).
    Now separated so we can call it twice for hybrid retrieval.
    """
    lang = detect_lang(query)
    query = correct_spelling(query, lang)

    paraphrases = generate_paraphrases(query, lang)
    paraphrases.extend(expand_agri_query(query))

    intent = compress_intent(query, lang)
    if intent and intent != query:
        paraphrases.append(intent)

    unique_queries = list(set(paraphrases))

    print(f"\n🔎 {debug_label} Processing {len(unique_queries)} variations for: '{query}'")
    print("-" * 60)

    all_hits = []

    for q in unique_queries:
        try:
            embedding = get_embedding(q)

            res = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=embedding,
                limit=top_k,
                with_payload=True
            )

            if res.points:
                print(f"   Query: '{q}'")
                for p in res.points:
                    doc = p.payload.get("doc_name", "Unknown")
                    cat = p.payload.get("category", "N/A")
                    chunk_lang = p.payload.get("language", "unknown")

                    print(f"     • [Score: {p.score:.4f}] {doc} ({cat}) [{chunk_lang}]")

                    all_hits.append({
                        "text": p.payload.get("text", ""),
                        "payload": p.payload,
                        "score": p.score,
                        "source_query": q
                    })

        except Exception as e:
            print(f"⚠️ Retrieval failed for '{q}': {e}")
            continue

    print("-" * 60)

    # Deduplication
    unique = {}
    for h in all_hits:
        t = h["text"]
        if t not in unique or h["score"] > unique[t]["score"]:
            unique[t] = h

    ranked = sorted(unique.values(), key=lambda x: x["score"], reverse=True)

    # Threshold filter
    ranked = [r for r in ranked if r["score"] >= MIN_SCORE_THRESHOLD]

    return ranked


def retrieve_chunks_hybrid(original_query, top_k=8):
    """
    NEW:
    If Urdu query -> translate to English and retrieve from both.
    Then merge, dedupe, rerank, keep top_k.
    """
    query_lang = detect_lang(original_query)

    # 1) Always retrieve using original query
    chunks_original = retrieve_chunks_single(
        original_query,
        top_k=top_k,
        debug_label="(Original)"
    )

    # 2) If Urdu, also retrieve using English translation
    chunks_translated = []
    translated_query = None

    if query_lang == "ur":
        translated_query = translate_ur_to_en(original_query)
        print(f"\n🌍 Urdu → English Retrieval Query: {translated_query}")

        chunks_translated = retrieve_chunks_single(
            translated_query,
            top_k=top_k,
            debug_label="(Translated)"
        )

    # 3) Merge + dedupe
    combined = chunks_original + chunks_translated

    unique = {}
    for h in combined:
        t = h["text"]
        if t not in unique or h["score"] > unique[t]["score"]:
            unique[t] = h

    merged = list(unique.values())

    # 4) Soft language boost
    for h in merged:
        h["score_boosted"] = language_boost_score(h, query_lang)

    # 5) Final rerank
    merged_sorted = sorted(merged, key=lambda x: x["score_boosted"], reverse=True)

    final_chunks = merged_sorted[:top_k]

    # --- FINAL DEBUG PRINT ---
    print(f"\n✅ FINAL HYBRID Top-{len(final_chunks)} Chunks Passed to GPT-4o:")
    if not final_chunks:
        print("   ❌ No chunks met the threshold.")
    else:
        for i, chunk in enumerate(final_chunks):
            doc = chunk["payload"].get("doc_name", "Unknown")
            lang = chunk["payload"].get("language", "unknown")
            print(f"   {i+1}. [Score: {chunk['score_boosted']:.4f}] {doc} [{lang}]")
            print(f"      Preview: \"{chunk['text'][:80].replace(chr(10), ' ')}...\"")
    print("=" * 60 + "\n")

    return final_chunks


# ============ MULTI-HOP (UNCHANGED, BUT USE HYBRID) ============

def generate_sub_query(query):
    words = query.split()
    return " ".join(words[:4])


def multi_hop_retrieval(query):
    print("\n🐰 Hop 1: Definition Search")
    sub_query = generate_sub_query(query)

    chunks_1 = retrieve_chunks_hybrid(sub_query, top_k=TOP_K)
    definition = generate_answer(sub_query, chunks_1, target_lang="en")

    grounding_text = definition[:500]

    print(f"\n🐰 Hop 2: Enriched Context Search")
    enriched_query = f"{query}. Context: {grounding_text}"

    chunks_2 = retrieve_chunks_hybrid(enriched_query, top_k=TOP_K)

    combined_map = {hash(c["text"]): c for c in chunks_1 + chunks_2}
    return list(combined_map.values())


# ============ ANSWERING ============

def generate_answer(original_query, chunks, target_lang="en"):
    if not chunks:
        return ("معلومات دستیاب نہیں۔" if target_lang == "ur" else "Information not available.")

    context_text = ""
    for c in chunks:
        doc = c['payload'].get('doc_name', 'Unknown')
        context_text += f"Source: {doc}\nContent: {c['text']}\n\n"

    if target_lang == "ur":
        system_prompt = (
            "You are an agricultural expert. You will receive context in English and Urdu. "
            "You must answer the user's question in clear, professional Urdu.\n\n"
            "IMPORTANT FORMATTING RULES:\n"
            "1) Use **Urdu Numerals** (۱, ۲, ۳) for lists followed by a dash (e.g., ۱- متن)\n"
            "2) Do NOT use English numbering (1. or 1-)\n"
            "3) Do NOT mention sources inside paragraphs\n"
            "4) At the very end, leave a blank line and list sources under '### حوالہ جات:'"
        )
    else:
        system_prompt = (
            "You are a helpful assistant. Answer ONLY using the provided Context. "
            "Do NOT cite sources inside the text sentences. "
            "At the very end, leave a blank line and list unique source names under '### Sources:'"
        )

    r = openai_client.chat.completions.create(
        model=GENERATION_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion:\n{original_query}"}
        ],
        temperature=0.3,
        max_tokens=800
    )

    return r.choices[0].message.content


# ============ PIPELINE ============

def rag_pipeline(query):
    print(f"\n💬 USER QUERY: {query}")

    query = normalize_query(query)
    query = normalize_intent(query)
    query = resolve_entity(query)
    query = expand_acronym_query(query)

    user_lang = detect_lang(query)

    # Multi-hop
    if is_multi_hop_question(query):
        chunks = multi_hop_retrieval(query)
        answer = generate_answer(query, chunks, target_lang=("ur" if user_lang == "ur" else "en"))
        return answer

    # Normal hybrid retrieval
    chunks = retrieve_chunks_hybrid(query, top_k=TOP_K)
    answer = generate_answer(query, chunks, target_lang=("ur" if user_lang == "ur" else "en"))
    return answer


# ============ TEST ============

if __name__ == "__main__":
    tests = [
        "What is PQNK?",
        "How to prune raddish?",
        "آلو کی کاشت کے لیے پانی دینے کا بہترین طریقہ کیا ہے؟",
    ]

    print("\n" + "=" * 50)
    print("🤖 AGRICULTURAL RAG CHATBOT (Hybrid Retrieval Enabled)")
    print("=" * 50 + "\n")

    for q in tests:
        console.print(f"[bold cyan]USER:[/bold cyan] {q}")
        raw_answer = rag_pipeline(q)
        console.print(f"[bold green]BOT:[/bold green]")
        console.print(Markdown(raw_answer))
        print("-" * 50 + "\n")




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








# =========== Areas TO Explore ==============
'''
apply category filter to the embeddings, filter also using the category the prompt 
is referring to.
Upload the complete name of the document, (To further scope it = focus on the exact 
pages if possible)

Meeting with an Alumni who worked on LLMs for his FYP:
-advised to use Google cloud vision for text extraction
-no current LLM's specifically specialised in Urdu training
-Told us to rework embeddings if time allows
-Will need multiple GPUs for open source option
-Results will not be accurate and can only get upto 70-75% accuracy
-GPU option will be expensive as well
-RAG accuracy can be improved over time

'''



# # ============ CLASSIFICATION LAYER ============
# # Define your physical folder names/categories here
# KNOWN_CATEGORIES = [
#     "crop_production", 
#     "animal_husbandry", 
#     "agricultural_history", 
#     "general"
# ]

# def classify_query_category(query, lang):
#     """
#     Determines which metadata filter to apply based on the user's intent.
#     """
#     # If the user asks a general question, we might not want to filter.
#     prompt = (
#         f"Classify this agricultural query into one of these categories: {KNOWN_CATEGORIES}.\n"
#         f"Query: {query}\n"
#         "Return ONLY the category name. If unsure or if it spans multiple, return 'general'."
#     )
    
#     try:
#         r = openai_client.chat.completions.create(
#             model="gpt-4o-mini",
#             messages=[{"role": "user", "content": prompt}],
#             temperature=0.0,
#             max_tokens=10
#         )
#         category = r.choices[0].message.content.strip().lower()
#         if category not in KNOWN_CATEGORIES:
#             return "general"
#         return category
#     except:
#         return "general"






# ============ UPGRADE FILTER WITH RETRIEVAL

# from qdrant_client.http import models as qmodels
# from sentence_transformers import CrossEncoder

# # Load a CrossEncoder model (Highly accurate, but slower than the bi-encoder)
# # This model looks at the Query and Document TOGETHER to score relevance.
# reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# def retrieve_chunks(query, top_k=TOP_K, verbose=False):
#     if not test_qdrant_connection(retries=1):
#         return []

#     lang = detect_lang(query)
#     query = correct_spelling(query, lang)
    
#     # 1. INTELLIGENT ROUTING
#     category_filter = classify_query_category(query, lang)
#     if verbose: print(f"📂 Identified Category: {category_filter}")

#     # Build Qdrant Filter
#     q_filter = None
#     if category_filter != "general":
#         q_filter = qmodels.Filter(
#             must=[
#                 qmodels.FieldCondition(
#                     key="category", # This matches your Qdrant Payload key
#                     match=qmodels.MatchValue(value=category_filter)
#                 )
#             ]
#         )

#     # 2. EXPANSION (Keep your existing logic)
#     search_queries = [query]
#     try:
#         search_queries.extend(expand_agri_query(query)[:1])
#     except:
#         pass

#     seen_texts = {}
    
#     # 3. RETRIEVAL (Bi-Encoder)
#     # We fetch MORE chunks than we need (top_k * 3) because we will filter them down later
#     initial_limit = top_k * 3 
    
#     for qtext in search_queries:
#         try:
#             emb = model.encode(qtext).tolist()
#             res = qdrant.query_points(
#                 collection_name=COLLECTION_NAME,
#                 query=emb,
#                 query_filter=q_filter, # <--- APPLYING THE FILTER HERE
#                 limit=initial_limit,
#                 with_payload=True
#             )
            
#             for p in res.points:
#                 text = p.payload.get("text", "")
#                 # Create a unique key
#                 key = hash(text)
                
#                 if key not in seen_texts:
#                     seen_texts[key] = {
#                         "text": text,
#                         "payload": p.payload,
#                         "initial_score": p.score,
#                         "query": qtext
#                     }
#         except Exception as e:
#             print(f"⚠️ Retrieval Error: {e}")
#             continue

#     candidates = list(seen_texts.values())

#     if not candidates:
#         return []

#     # 4. RE-RANKING (Cross-Encoder) - The "Perfecting" Step
#     # We pass [Query, Document] pairs to the model.
#     pairs = [[query, c["text"]] for c in candidates]
    
#     try:
#         cross_scores = reranker.predict(pairs)
        
#         # Attach new scores
#         for i, c in enumerate(candidates):
#             c["rerank_score"] = cross_scores[i]

#         # Sort by the new, highly accurate Cross-Encoder score
#         ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        
#         # Filter by Re-ranker threshold (usually numbers are different, e.g., > 0 or > -2)
#         # You will need to test what threshold works, start with very low.
#         filtered = [r for r in ranked if r["rerank_score"] > -4.0] 

#     except Exception as e:
#         print(f"⚠️ Reranker failed, falling back to vector score: {e}")
#         ranked = sorted(candidates, key=lambda x: x["initial_score"], reverse=True)
#         filtered = ranked

#     # 5. FINAL SELECTION
#     final_chunks = filtered[:top_k]

#     if verbose:
#         print(f"✅ Retrieved {len(final_chunks)} chunks after reranking.")
#         for c in final_chunks:
#             print(f"   Score: {c.get('rerank_score', c['initial_score']):.4f} | Doc: {c['payload'].get('doc_name')}")

#     return final_chunks