# import os
# import glob
# from uuid import uuid4
# from tqdm import tqdm
# from openai import OpenAI
# from qdrant_client import QdrantClient
# from qdrant_client.http import models
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langdetect import detect, LangDetectException


# # ================= CONFIGURATION =================

import os
import glob
from uuid import uuid4
from tqdm import tqdm
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langdetect import detect, DetectorFactory, LangDetectException

# ================= CONFIGURATION =================

# 🔒 Make language detection deterministic
DetectorFactory.seed = 0

# 🔑 API KEYS (Ideally set these in your environment variables)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# 📂 DATA SOURCE
ROOT_DATA_DIR = "./text_pdfs"

# ⚙️ SETTINGS
COLLECTION_NAME = "pqnk_v2"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE = 1536

# Chunking Settings
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ================= INITIALIZATION =================

openai_client = OpenAI(api_key=OPENAI_API_KEY)
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", "۔", ".", " ", ""]
)

# ================= FUNCTIONS =================

def ensure_collection_exists():
    collections = qdrant_client.get_collections()
    exists = any(c.name == COLLECTION_NAME for c in collections.collections)

    if not exists:
        print(f"📦 Creating collection '{COLLECTION_NAME}'...")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE
            )
        )
    else:
        print(f"✅ Collection '{COLLECTION_NAME}' found.")


def get_openai_embedding(text):
    text = text.replace("\n", " ")
    return openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL
    ).data[0].embedding


def detect_language_from_text(text):
    """
    Detect language from content.
    Returns: 'en', 'ur', or 'unknown'
    """
    try:
        lang = detect(text)
        if lang in ("en", "ur"):
            return lang
        return "unknown"
    except LangDetectException:
        return "unknown"


def parse_file_metadata(filepath):
    """
    Extract category and filename only.
    Language is detected from content, not filename.
    """
    path_parts = os.path.normpath(filepath).split(os.sep)
    category = path_parts[-2]
    filename = path_parts[-1]
    return category, filename


def process_and_upload():
    ensure_collection_exists()

    search_path = os.path.join(ROOT_DATA_DIR, "**", "*.txt")
    files = glob.glob(search_path, recursive=True)

    print(f"📂 Found {len(files)} files to process.")

    total_chunks = 0

    for filepath in tqdm(files, desc="Processing Files"):
        try:
            category, filename = parse_file_metadata(filepath)

            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                continue

            # 1️⃣ Create chunks
            chunks = text_splitter.split_text(content)

            points_batch = []

            # 2️⃣ Process each chunk independently
            for i, chunk_text in enumerate(chunks):
                language = detect_language_from_text(chunk_text)
                vector = get_openai_embedding(chunk_text)

                payload = {
                    "text": chunk_text,
                    "doc_name": filename,
                    "category": category,
                    "language": language,
                    "chunk_id": i,
                    "source_path": filepath
                }

                points_batch.append(
                    models.PointStruct(
                        id=str(uuid4()),
                        vector=vector,
                        payload=payload
                    )
                )

            # 3️⃣ Upload to Qdrant
            if points_batch:
                qdrant_client.upsert(
                    collection_name=COLLECTION_NAME,
                    points=points_batch
                )
                total_chunks += len(points_batch)

        except Exception as e:
            print(f"\n❌ Error processing {filepath}: {e}")

    print(f"\n🎉 Done! Uploaded {total_chunks} chunks to Qdrant.")


# ================= RUN =================

if __name__ == "__main__":
    process_and_upload()
