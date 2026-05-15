"""
ingest_video_txts.py
====================
Bulk-ingests video transcript .txt files into both Qdrant (OpenAI embeddings)
and PostgreSQL (sentence-transformer embeddings).

Expected folder layout (subfolder names can be anything):
    video_texts/
      ├── <any_name>/
      │     ├── animal_husbandry/
      │     │     ├── lecture_urdu.txt
      │     │     └── lecture_english.txt
      │     └── soil_science/
      │           └── soil_basics.txt
      ├── <any_other_name>/
      │     └── crop_production/
      │           └── wheat_yield.txt
      └── ...

Run from the project root:
    python ingest_video_txts.py
"""

import os
import re
import sys
from uuid import uuid4

from tqdm import tqdm

# ── Load .env ──────────────────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"), override=True)
except ImportError:
    pass

# ── Configuration ──────────────────────────────────────────────────────────────
ROOT_DIR        = os.path.join(os.path.dirname(__file__), "videos_texts")

# Qdrant
QDRANT_URL= os.getenv("QDRANT_URL", "")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME = "pqnk_v2"
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_SIZE     = 1536
CHUNK_SIZE      = 1000
CHUNK_OVERLAP   = 200

# PostgreSQL
DB_NAME = os.getenv("DB_NAME", "pqnk_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")

# Source label stored in Qdrant payload so these can be filtered
# separately from documents / images later.
SOURCE_LABEL = "video"

# Files whose stem ends with these suffixes are intermediate pipeline
# artefacts — skip them.
SKIP_SUFFIXES = ("_final",)

# Minimum character count — Whisper sometimes produces near-empty files
# for silent segments; skip those.
MIN_CHARS = 50


# ── Lazy client initialisation ─────────────────────────────────────────────────
_qdrant      = None
_openai      = None
_pg_conn     = None
_st_model    = None
_splitter    = None


def get_qdrant():
    global _qdrant
    if _qdrant is None:
        from qdrant_client import QdrantClient
        _qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return _qdrant


def get_openai():
    global _openai
    if _openai is None:
        from openai import OpenAI
        _openai = OpenAI(api_key=OPENAI_API_KEY)
    return _openai


def get_pg():
    global _pg_conn
    if _pg_conn is None or _pg_conn.closed:
        import psycopg2
        _pg_conn = psycopg2.connect(
            database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST
        )
    return _pg_conn


def get_st_model():
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        print("[Init] Loading sentence-transformer model...")
        _st_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        print("[Init] ✅ Model loaded.")
    return _st_model


def get_splitter():
    global _splitter
    if _splitter is None:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        _splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", "۔", ".", " ", ""],
        )
    return _splitter


# ── Text helpers ───────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Fix common mojibake left by PDF/Whisper extraction."""
    replacements = {
        "â€¢": "•", "â€“": "–", "â€”": "—", "â€˜": "‘", "â€™": "’",
        "â€œ": "“", "â€ ": "”", "â€¦": "…", "â€“": "-",
        "â†’": "→", "Â": "",
    }
    for bad, good in replacements.items():
        text = text.replace(bad, good)
    return text


def detect_language(text: str) -> str:
    """
    Chunk-level language detection.
    Uses character-ratio heuristic first (fast + reliable for Urdu/English),
    then falls back to langdetect.
    """
    urdu_chars    = len(re.findall(r"[\u0600-\u06FF]", text))
    english_chars = len(re.findall(r"[A-Za-z]", text))
    total         = len(text)

    if total == 0:
        return "unknown"
    if urdu_chars / total > 0.3:
        return "ur"
    if english_chars / total > 0.3:
        return "en"

    try:
        from langdetect import detect, LangDetectException
        lang = detect(text)
        if "ur" in lang:
            return "ur"
        if "en" in lang:
            return "en"
    except Exception:
        pass

    return "unknown"


# ── Qdrant helpers ─────────────────────────────────────────────────────────────

def ensure_qdrant_collection():
    from qdrant_client.http import models
    qd = get_qdrant()
    existing = [c.name for c in qd.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"[Qdrant] Creating collection '{COLLECTION_NAME}'...")
        qd.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(
                size=VECTOR_SIZE,
                distance=models.Distance.COSINE,
            ),
        )
    else:
        print(f"[Qdrant] ✅ Collection '{COLLECTION_NAME}' already exists.")


def embed_and_upsert_qdrant(chunks: list, filename: str, category: str, filepath: str):
    """Embed every chunk via OpenAI and upsert into Qdrant."""
    from qdrant_client.http import models

    openai_client = get_openai()
    qd            = get_qdrant()
    points        = []

    for i, chunk_text in enumerate(chunks):
        lang   = detect_language(chunk_text)
        vector = openai_client.embeddings.create(
            input=[chunk_text.replace("\n", " ")],
            model=EMBEDDING_MODEL,
        ).data[0].embedding

        points.append(
            models.PointStruct(
                id=str(uuid4()),
                vector=vector,
                payload={
                    "text":        chunk_text,
                    "doc_name":    filename,
                    "category":    category,
                    "language":    lang,
                    "chunk_id":    i,
                    "source_path": filepath,
                    "source":      SOURCE_LABEL,   # ← "video" label
                },
            )
        )

    if points:
        qd.upsert(collection_name=COLLECTION_NAME, points=points)

    return len(points)


# ── PostgreSQL helpers ─────────────────────────────────────────────────────────

def already_in_postgres(filename: str, category: str) -> bool:
    """
    Duplicate guard: returns True if this exact filename + category
    combination already exists in the content table.
    Prevents re-ingestion when the script is run multiple times.
    """
    conn = get_pg()
    cur  = conn.cursor()
    cur.execute(
        "SELECT 1 FROM content WHERE filename = %s AND category = %s LIMIT 1",
        (filename, category),
    )
    exists = cur.fetchone() is not None
    cur.close()
    return exists


def ingest_to_postgres(chunks: list, filename: str, category: str):
    """Insert content metadata + chunk embeddings into PostgreSQL."""
    conn  = get_pg()
    cur   = conn.cursor()
    model = get_st_model()

    # Determine file-level language from filename convention
    # (pipeline_service.py names files *_urdu.txt / *_english.txt)
    if "urdu" in filename.lower():
        file_lang = "ur"
    elif "english" in filename.lower() or "eng" in filename.lower():
        file_lang = "en"
    else:
        file_lang = "unknown"

    try:
        cur.execute(
            "INSERT INTO content (filename, category, language) VALUES (%s, %s, %s) RETURNING content_id",
            (filename, category, file_lang),
        )
        content_id = cur.fetchone()[0]

        for i, chunk in enumerate(chunks):
            lang      = detect_language(chunk)
            embedding = model.encode(chunk).tolist()
            cur.execute(
                "INSERT INTO vectors (content_id, chunk_text, chunk_index, embedding) VALUES (%s, %s, %s, %s)",
                (content_id, chunk, i, embedding),
            )

        conn.commit()
    except Exception as exc:
        conn.rollback()
        raise exc
    finally:
        cur.close()


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_txt_files(root: str) -> list:
    """
    Walk root → subfolder (any name) → category folder → *.txt files.
    Returns a list of (filepath, category, subfolder_name) tuples.

    Skips:
      - Files whose stem ends with any SKIP_SUFFIXES (e.g. '_final')
      - Non-.txt files
      - Category folders that are themselves the direct children of root
        (enforces the two-level depth: root/subfolder/category/file.txt)
    """
    found = []

    if not os.path.isdir(root):
        print(f"[Discovery] ❌ Root folder not found: {root}")
        sys.exit(1)

    # Level 1: subfolder (any random name)
    for subfolder_name in sorted(os.listdir(root)):
        subfolder_path = os.path.join(root, subfolder_name)
        if not os.path.isdir(subfolder_path):
            continue

        # Level 2: category folder
        for category_name in sorted(os.listdir(subfolder_path)):
            category_path = os.path.join(subfolder_path, category_name)
            if not os.path.isdir(category_path):
                continue

            # Level 3: txt files
            for fname in sorted(os.listdir(category_path)):
                if not fname.lower().endswith(".txt"):
                    continue

                stem = os.path.splitext(fname)[0]
                if any(stem.endswith(s) for s in SKIP_SUFFIXES):
                    continue

                found.append((
                    os.path.join(category_path, fname),
                    category_name,
                    subfolder_name,
                ))

    return found


# ── Main orchestrator ──────────────────────────────────────────────────────────

def run():
    print("=" * 60)
    print("  PQNK Video TXT Bulk Ingestion")
    print(f"  Root : {ROOT_DIR}")
    print(f"  Target collection : {COLLECTION_NAME}")
    print("=" * 60)

    # 1. Discover all eligible txt files
    all_files = discover_txt_files(ROOT_DIR)
    if not all_files:
        print("[Discovery] ⚠️  No .txt files found. Check your folder structure.")
        return

    print(f"\n[Discovery] Found {len(all_files)} .txt file(s) across all subfolders.\n")

    # 2. Make sure Qdrant collection exists
    ensure_qdrant_collection()

    # 3. Load sentence-transformer once (avoids reloading per file)
    get_st_model()

    # 4. Process each file
    splitter = get_splitter()

    total_files_processed = 0
    total_files_skipped   = 0   # duplicates
    total_files_too_short = 0   # empty / near-empty Whisper output
    total_chunks_qdrant   = 0
    total_chunks_pg       = 0

    for filepath, category, subfolder in tqdm(all_files, desc="Ingesting files"):
        filename = os.path.basename(filepath)
        tag      = f"{subfolder}/{category}/{filename}"

        try:
            # ── Read & clean ──────────────────────────────────────────
            with open(filepath, "r", encoding="utf-8", errors="ignore") as fh:
                raw = clean_text(fh.read().strip())

            # ── Guard 1: too short (silent / corrupt Whisper output) ──
            if len(raw) < MIN_CHARS:
                tqdm.write(f"  [SKIP - too short] {tag}  ({len(raw)} chars)")
                total_files_too_short += 1
                continue

            # ── Guard 2: duplicate in PostgreSQL ─────────────────────
            if already_in_postgres(filename, category):
                tqdm.write(f"  [SKIP - duplicate] {tag}")
                total_files_skipped += 1
                continue

            # ── Chunk ─────────────────────────────────────────────────
            chunks = splitter.split_text(raw)
            if not chunks:
                tqdm.write(f"  [SKIP - no chunks] {tag}")
                continue

            tqdm.write(f"  [OK] {tag}  →  {len(chunks)} chunk(s)")

            # ── Ingest to Qdrant ──────────────────────────────────────
            n_qdrant = embed_and_upsert_qdrant(chunks, filename, category, filepath)
            total_chunks_qdrant += n_qdrant

            # ── Ingest to PostgreSQL ──────────────────────────────────
            ingest_to_postgres(chunks, filename, category)
            total_chunks_pg += len(chunks)

            total_files_processed += 1

        except Exception as exc:
            tqdm.write(f"  [ERROR] {tag}: {exc}")

    # 5. Summary
    print("\n" + "=" * 60)
    print("  Ingestion Complete")
    print("=" * 60)
    print(f"  Files processed       : {total_files_processed}")
    print(f"  Files skipped (dupe)  : {total_files_skipped}")
    print(f"  Files skipped (short) : {total_files_too_short}")
    print(f"  Chunks → Qdrant       : {total_chunks_qdrant}")
    print(f"  Chunks → PostgreSQL   : {total_chunks_pg}")
    print("=" * 60)


if __name__ == "__main__":
    run()
