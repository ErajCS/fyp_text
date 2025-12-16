import os
import re
import psycopg2
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from langdetect import detect

# --- CONFIG ---
BASE_DIR = r"text_pdfs"  # Matches your folder structure
DB_NAME = "pqnk_db"
DB_USER = "postgres"
DB_PASS = "admin123"  # <--- CHANGE THIS

# Load Model
print("Loading Model...")
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

def get_db_connection():
    return psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")

# --- YOUR TEXT PROCESSING FUNCTIONS ---
def clean_text(text):
    replacements = {
        "â€¢": "•", "â€“": "–", "â€”": "—", "â€˜": "‘", "â€™": "’",
        "â€œ": "“", "â€ ": "”", "â€¦": "…", "â€“": "-",
        "â†’": "→", "Â": "", "~": "~", "=": "=", "(": "(", ")": ")",
        "≈": "≈", "!": "!", ":": ":"
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk.strip())
        start += (chunk_size - overlap)
    return chunks

def detect_language_per_chunk(text):
    urdu_chars = len(re.findall(r'[\u0600-\u06FF]', text))
    english_chars = len(re.findall(r'[A-Za-z]', text))
    total_chars = len(text)
    if total_chars == 0: return 'unknown'
    
    if (urdu_chars / total_chars) > 0.3: return 'ur'
    if (english_chars / total_chars) > 0.3: return 'en'
    try:
        lang = detect(text)
        return 'ur' if 'ur' in lang else 'en' if 'en' in lang else 'unknown'
    except:
        return 'unknown'

def ingest_data():
    conn = get_db_connection()
    cur = conn.cursor()

    # Iterate through Categories
    for category in os.listdir(BASE_DIR):
        cat_path = os.path.join(BASE_DIR, category)
        if not os.path.isdir(cat_path): continue
        
        print(f"\n📂 Processing Category: {category}")
        
        for file in os.listdir(cat_path):
            if not file.endswith(".txt"): continue
            
            file_path = os.path.join(cat_path, file)
            print(f"  -> Processing {file}...")

            try:
                # Read File
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = clean_text(f.read().strip())
                
                if len(text) < 50: continue

                # Insert into Content Table (Metadata)
                # We assume language is mixed, so we mark file generally, but chunks are specific
                file_lang = "ur" if "urdu" in file.lower() else "en"
                
                cur.execute("""
                    INSERT INTO content (filename, category, language)
                    VALUES (%s, %s, %s) RETURNING content_id
                """, (file, category, file_lang))
                content_id = cur.fetchone()[0]

                # Process Chunks
                chunks = chunk_text(text)
                
                for i, chunk in enumerate(chunks):
                    lang = detect_language_per_chunk(chunk)
                    
                    # Generate Embedding
                    embedding = model.encode(chunk).tolist()
                    
                    # Insert into Vectors Table
                    cur.execute("""
                        INSERT INTO vectors (content_id, chunk_text, chunk_index, embedding)
                        VALUES (%s, %s, %s, %s)
                    """, (content_id, chunk, i, embedding))
                
                conn.commit()
                
            except Exception as e:
                print(f"❌ Error processing {file}: {e}")
                conn.rollback()

    cur.close()
    conn.close()
    print("\n🎉 All data ingested into PostgreSQL successfully!")

if __name__ == "__main__":
    ingest_data()