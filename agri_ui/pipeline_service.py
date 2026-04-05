"""
PQNK Automation Pipeline Orchestrator
======================================
Triggered by the Flask upload endpoint in app.py after a file is saved.
Runs as a background thread so the HTTP response is returned immediately.

Scenarios:
  - document  → extraction.py (text extraction + OCR)
               → [NEW] extracting_images_from_pdfs.py (extract embedded images)
                 → detection_of_lang_and_renaming.py (per extracted image)
                 → translating_images.py (GPT-4o-mini bilingual description)
               → ingest_data.py + ingest_to_postgres.py (text + image txts)
  - image     → detection_of_lang_and_renaming.py → translating_images.py
                → ingest_data.py + ingest_to_postgres.py
  - video     → local_whisper.py → local_refining.py → translating_video.py
                → ingest_data.py + ingest_to_postgres.py
  - audio     → [NEW] Faster-Whisper transcription (no ffmpeg, already MP3)
                → local_refining.py → translating_video.py
                → ingest_data.py + ingest_to_postgres.py
                NOTE: audio is INVISIBLE in the repository UI (AI-only ingestion)
"""

import os
import sys
import glob
import threading
import subprocess
import pathlib
from uuid import uuid4

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "..", ".env"))
except Exception:
    pass

# Import drive service for async sync
import drive_service as _drive_svc

# ── Path bootstrap so we can import root-level scripts ──────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# ── Pipeline working directories ─────────────────────────────────────────────
# All intermediate files live here (txt, mp3, refined transcripts, etc.)
PIPELINE_WORK_DIR = os.path.join(REPO_ROOT, "pipeline_workspace")
PDF_WORK_DIR      = os.path.join(PIPELINE_WORK_DIR, "documents")
IMAGE_WORK_DIR    = os.path.join(PIPELINE_WORK_DIR, "images")
VIDEO_WORK_DIR    = os.path.join(PIPELINE_WORK_DIR, "videos")
AUDIO_WORK_DIR    = os.path.join(PIPELINE_WORK_DIR, "audios")
FFMPEG_PATH       = os.path.join(REPO_ROOT, "test_video", "ffmpeg.exe")  # same as local_whisper.py

for _d in [PIPELINE_WORK_DIR, PDF_WORK_DIR, IMAGE_WORK_DIR, VIDEO_WORK_DIR, AUDIO_WORK_DIR]:
    os.makedirs(_d, exist_ok=True)

# ── Qdrant / OpenAI config (from .env) ───────────────────────────────────────
QDRANT_URL        = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY", "")
OPENAI_API_KEY    = os.getenv("OPENAI_API_KEY", "")
COLLECTION_NAME   = "pqnk_v2"
EMBEDDING_MODEL   = "text-embedding-3-small"
CHUNK_SIZE        = 1000
CHUNK_OVERLAP     = 200

# ── PostgreSQL config (from .env) ─────────────────────────────────────────────
DB_NAME = os.getenv("DB_NAME", "pqnk_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")


# ==============================================================================
# SECTION 1 — QDRANT INGESTION  (mirrors ingest_data.py logic)
# ==============================================================================

def _ingest_txt_files_to_qdrant(txt_files: list, category: str, source_label: str) -> int:
    """
    Embed a list of .txt file paths and upsert them into Qdrant.
    Returns total chunk count uploaded.
    """
    from openai import OpenAI
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langdetect import detect, DetectorFactory, LangDetectException
    DetectorFactory.seed = 0

    openai_client  = OpenAI(api_key=OPENAI_API_KEY)
    qdrant_client  = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    splitter       = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "۔", ".", " ", ""]
    )

    # Ensure collection exists
    existing = [c.name for c in qdrant_client.get_collections().collections]
    if COLLECTION_NAME not in existing:
        print(f"[Pipeline] Creating Qdrant collection '{COLLECTION_NAME}'")
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
        )

    total_chunks = 0

    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read().strip()
            if not content:
                continue

            chunks = splitter.split_text(content)
            points = []
            for i, chunk_text in enumerate(chunks):
                try:
                    lang = detect(chunk_text)
                    if lang not in ("en", "ur"):
                        lang = "unknown"
                except LangDetectException:
                    lang = "unknown"

                vector = openai_client.embeddings.create(
                    input=[chunk_text.replace("\n", " ")],
                    model=EMBEDDING_MODEL
                ).data[0].embedding

                points.append(models.PointStruct(
                    id=str(uuid4()),
                    vector=vector,
                    payload={
                        "text":        chunk_text,
                        "doc_name":    os.path.basename(fpath),
                        "category":    category,
                        "language":    lang,
                        "chunk_id":    i,
                        "source_path": fpath,
                        "source":      source_label,
                    }
                ))

            if points:
                qdrant_client.upsert(collection_name=COLLECTION_NAME, points=points)
                total_chunks += len(points)

        except Exception as exc:
            print(f"[Pipeline] ⚠️  Qdrant ingest error for {fpath}: {exc}")

    return total_chunks


# ==============================================================================
# SECTION 2 — POSTGRESQL INGESTION  (mirrors ingest_to_postgres.py logic)
# ==============================================================================

def _ingest_txt_files_to_postgres(txt_files: list, category: str):
    """
    Embed a list of .txt file paths using sentence-transformers and insert
    into the PostgreSQL `content` + `vectors` tables.
    """
    import re
    import psycopg2
    from sentence_transformers import SentenceTransformer
    from langdetect import detect

    model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    def _chunk(text, size=500, overlap=100):
        words = text.split()
        out   = []
        i = 0
        while i < len(words):
            out.append(" ".join(words[i:i+size]))
            i += size - overlap
        return out

    def _lang_of_chunk(text):
        ur = len(re.findall(r"[\u0600-\u06FF]", text))
        en = len(re.findall(r"[A-Za-z]", text))
        tot = len(text)
        if tot == 0: return "unknown"
        if ur/tot > 0.3: return "ur"
        if en/tot > 0.3: return "en"
        try:
            l = detect(text)
            return "ur" if "ur" in l else "en" if "en" in l else "unknown"
        except Exception:
            return "unknown"

    try:
        conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
        cur  = conn.cursor()
    except Exception as exc:
        print(f"[Pipeline] ⚠️  PostgreSQL connection failed: {exc}")
        return

    for fpath in txt_files:
        try:
            with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                raw = f.read().strip()
            if len(raw) < 50:
                continue

            fname     = os.path.basename(fpath)
            file_lang = "ur" if "urdu" in fname.lower() else "en"

            cur.execute(
                "INSERT INTO content (filename, category, language) VALUES (%s, %s, %s) RETURNING content_id",
                (fname, category, file_lang)
            )
            content_id = cur.fetchone()[0]

            for i, chunk in enumerate(_chunk(raw)):
                lang      = _lang_of_chunk(chunk)
                embedding = model.encode(chunk).tolist()
                cur.execute(
                    "INSERT INTO vectors (content_id, chunk_text, chunk_index, embedding) VALUES (%s, %s, %s, %s)",
                    (content_id, chunk, i, embedding)
                )

            conn.commit()

        except Exception as exc:
            print(f"[Pipeline] ⚠️  PostgreSQL ingest error for {fpath}: {exc}")
            try:
                conn.rollback()
            except Exception:
                pass

    cur.close()
    conn.close()


def _sync_to_drive_step(file_path: str, file_type: str, category: str, original_name: str, item_id: int):
    """
    Uploads the file to Google Drive and updates the PG database with the ID/link.
    This effectively moves the slow Drive upload out of the main request thread.
    """
    if not item_id:
        return

    print(f"[Pipeline] ☁️  Syncing to Google Drive: {original_name or os.path.basename(file_path)}...")
    
    # Map file_type to Folder Name (same as in app.py)
    type_folder_map = {"document": "PDFs", "image": "Images", "video": "Videos", "audio": "Audios"}
    drive_path = [type_folder_map.get(file_type, "General"), category]

    try:
        drive_result = _drive_svc.upload_file(
            local_path     = file_path,
            filename       = original_name or os.path.basename(file_path),
            mime_type      = _drive_svc.get_mime_type(original_name or os.path.basename(file_path)),
            subfolder_path = drive_path
        )
        file_id   = drive_result.get("file_id")
        view_link = drive_result.get("view_link")

        if file_id:
            import psycopg2
            conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
            cur  = conn.cursor()
            cur.execute(
                "UPDATE resources SET drive_file_id = %s, drive_view_link = %s WHERE id = %s",
                (file_id, view_link, item_id)
            )
            conn.commit()
            cur.close()
            conn.close()
            print(f"[Pipeline] ✅ Drive Sync Complete for ID {item_id}")
    except Exception as exc:
        print(f"[Pipeline] ⚠️  Drive Sync failed: {exc}")


# ==============================================================================
# SECTION 3 — DOCUMENT SCENARIO
# ==============================================================================

def _run_document_pipeline(file_path: str, category: str) -> int:
    """
    Runs extraction.py → qdrant + postgres.
    Also extracts any embedded images from the PDF and processes them through
    the full image pipeline (lang detection → GPT-4o-mini description → ingest).
    Returns total chunk count indexed.
    """
    print(f"\n[Pipeline] 📄 DOCUMENT: {os.path.basename(file_path)}")

    # Step 1 — OCR + translate (uses VisionPDFTranslator from extraction.py)
    from extraction import VisionPDFTranslator

    work_dir = os.path.join(PDF_WORK_DIR, category)
    os.makedirs(work_dir, exist_ok=True)

    # Copy the uploaded file into the working directory so extraction.py
    # writes its txt outputs alongside the pdf.
    import shutil
    dest_pdf = os.path.join(work_dir, os.path.basename(file_path))
    if not os.path.exists(dest_pdf):
        shutil.copy2(file_path, dest_pdf)

    processor = VisionPDFTranslator(work_dir)
    processor.process_pdf(pathlib.Path(dest_pdf))

    # Step 2 — Collect the generated txt files from text extraction
    base = pathlib.Path(dest_pdf).stem
    txt_files = [
        str(pathlib.Path(work_dir) / f"{base}_urdu.txt"),
        str(pathlib.Path(work_dir) / f"{base}_english.txt"),
    ]
    txt_files = [t for t in txt_files if os.path.exists(t)]

    if not txt_files:
        print(f"[Pipeline] ⚠️  No txt files generated for {os.path.basename(file_path)}. Check extraction logs.")

    if txt_files:
        print(f"[Pipeline] Generated {len(txt_files)} txt file(s): {[os.path.basename(t) for t in txt_files]}")

    # Step 3 — Ingest text content to Qdrant + PostgreSQL
    total_chunks = 0
    if txt_files:
        total_chunks += _ingest_txt_files_to_qdrant(txt_files, category, "document")
        _ingest_txt_files_to_postgres(txt_files, category)

    # ── STEP 4 — Extract images embedded in the PDF and run image pipeline ──
    total_chunks += _run_pdf_image_sub_pipeline(dest_pdf, work_dir, category)

    return total_chunks


def _run_pdf_image_sub_pipeline(pdf_path: str, work_dir: str, category: str) -> int:
    """
    Extracts all meaningful images embedded in *pdf_path* using the same logic
    as extracting_images_from_pdfs.py, then passes each extracted image through
    the standard image pipeline:
      1. detect_language + rename  (_eng / _urdu suffix)
      2. GPT-4o-mini bilingual description → .txt files
      3. Ingest txt files to Qdrant + PostgreSQL
    Returns total chunk count from image descriptions.
    """
    try:
        from extracting_images_from_pdfs import (
            extract_images_from_pdf,
            load_logo_templates,
        )
        from detection_of_lang_and_renaming import detect_language, rename_image
        from translating_images import ImageDescriber
        from PIL import Image
    except ImportError as e:
        print(f"[Pipeline] ⚠️  Image sub-pipeline import failed ({e}). Skipping PDF image extraction.")
        return 0

    # Dedicated subfolder so extracted images don't collide with document txt files
    img_out_dir = os.path.join(work_dir, f"{pathlib.Path(pdf_path).stem}_images")
    os.makedirs(img_out_dir, exist_ok=True)

    print(f"[Pipeline] 🖼️  Extracting embedded images from PDF: {os.path.basename(pdf_path)}")

    # Load logo templates to filter decorative/logo images (may be empty if folder absent)
    logo_template_dir = os.path.join(REPO_ROOT, "text_pdfs", "logo_templates")
    logo_templates = load_logo_templates(logo_template_dir)

    try:
        extract_images_from_pdf(pdf_path, img_out_dir, logo_templates)
    except Exception as e:
        print(f"[Pipeline] ⚠️  Image extraction from PDF failed: {e}")
        return 0

    # Collect extracted image files
    extracted_images = [
        f for f in pathlib.Path(img_out_dir).iterdir()
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    if not extracted_images:
        print(f"[Pipeline] ℹ️  No meaningful images found inside PDF.")
        return 0

    print(f"[Pipeline] 🖼️  Found {len(extracted_images)} image(s) inside PDF — processing each...")

    total_image_chunks = 0

    for img_path in extracted_images:
        try:
            # Step A — Language detection + rename to *_eng.ext or *_urdu.ext
            dest_img = str(img_path)
            pil_img  = Image.open(dest_img).convert("RGB")
            lang     = detect_language(pil_img)

            if lang in ["eng", "urdu"]:
                rename_image(dest_img, lang)
                name, ext  = os.path.splitext(dest_img)
                dest_img   = f"{name}_{lang}{ext}"
            else:
                # Default to English when language is ambiguous
                print(f"[Pipeline] ℹ️  Language ambiguous for {img_path.name}, defaulting to 'eng'")
                rename_image(dest_img, "eng")
                name, ext  = os.path.splitext(dest_img)
                dest_img   = f"{name}_eng{ext}"

            # Step B — GPT-4o-mini bilingual vision description → *.txt files
            describer = ImageDescriber(img_out_dir)
            describer.process_image(pathlib.Path(dest_img))

            # Step C — Collect generated txt files for this image
            stem       = pathlib.Path(dest_img).stem
            img_base   = stem.replace("_urdu", "").replace("_eng", "")
            img_txts   = [str(p) for p in pathlib.Path(img_out_dir).glob(f"{img_base}*.txt")]

            if img_txts:
                total_image_chunks += _ingest_txt_files_to_qdrant(img_txts, category, "document_image")
                _ingest_txt_files_to_postgres(img_txts, category)
                print(f"[Pipeline]   ✅ Indexed {len(img_txts)} txt(s) for image: {img_path.name}")
            else:
                print(f"[Pipeline]   ⚠️  No txt generated for image: {img_path.name}")

        except Exception as e:
            print(f"[Pipeline]   ⚠️  Failed processing image {img_path.name}: {e}")
            continue

    print(f"[Pipeline] ✅ PDF image sub-pipeline complete: {total_image_chunks} chunks from {len(extracted_images)} image(s).")
    return total_image_chunks


# ==============================================================================
# SECTION 4 — IMAGE SCENARIO
# ==============================================================================

def _run_image_pipeline(file_path: str, category: str) -> int:
    """
    Runs detection_of_lang_and_renaming.py → translating_images.py → qdrant + postgres.
    Returns chunk count indexed.
    """
    print(f"\n[Pipeline] 🖼️  IMAGE: {os.path.basename(file_path)}")

    from detection_of_lang_and_renaming import detect_language, rename_image
    from translating_images import ImageDescriber
    from PIL import Image

    work_dir = os.path.join(IMAGE_WORK_DIR, category)
    os.makedirs(work_dir, exist_ok=True)

    # Copy uploaded image to work dir
    import shutil
    dest_img = os.path.join(work_dir, os.path.basename(file_path))
    if not os.path.exists(dest_img):
        shutil.copy2(file_path, dest_img)

    # Step 1 — Detect language and rename
    try:
        pil_img = Image.open(dest_img).convert("RGB")
        lang = detect_language(pil_img)
        if lang in ["eng", "urdu"]:
            rename_image(dest_img, lang)
            # Update dest_img path to the renamed file
            name, ext = os.path.splitext(dest_img)
            dest_img = f"{name}_{lang}{ext}"
        else:
            print(f"[Pipeline] ⚠️  Language not detected for {os.path.basename(dest_img)}, treating as 'eng'")
            rename_image(dest_img, "eng")
            name, ext = os.path.splitext(dest_img)
            dest_img = f"{name}_eng{ext}"
    except Exception as exc:
        print(f"[Pipeline] ⚠️  Language detection skipped: {exc}")

    # Step 2 — GPT-4o-mini vision → bilingual txt files
    try:
        describer = ImageDescriber(work_dir)
        describer.process_image(pathlib.Path(dest_img))
    except Exception as exc:
        print(f"[Pipeline] ⚠️  Image description failed: {exc}")
        return 0

    # Step 3 — Collect generated txt files
    stem    = pathlib.Path(dest_img).stem
    base    = stem.replace("_urdu", "").replace("_eng", "")
    txt_files = []
    for candidate in pathlib.Path(work_dir).glob(f"{base}*.txt"):
        txt_files.append(str(candidate))

    if not txt_files:
        print(f"[Pipeline] ⚠️  No txt files generated for image.")
        return 0

    print(f"[Pipeline] Generated {len(txt_files)} txt file(s): {[os.path.basename(t) for t in txt_files]}")

    # Step 4 — Ingest
    chunks = _ingest_txt_files_to_qdrant(txt_files, category, "image")
    _ingest_txt_files_to_postgres(txt_files, category)

    return chunks


# ==============================================================================
# SECTION 5 — VIDEO SCENARIO
# ==============================================================================

def _run_video_pipeline(file_path: str, category: str) -> int:
    """
    Runs local_whisper.py logic → local_refining.py → translating_video.py → qdrant + postgres.
    Returns chunk count indexed.
    """
    print(f"\n[Pipeline] 🎬 VIDEO: {os.path.basename(file_path)}")

    import subprocess
    from faster_whisper import WhisperModel
    from local_refining import refine_single_file, is_english
    from translating_video import TXTDeepTranslator

    work_dir = os.path.join(VIDEO_WORK_DIR, category)
    os.makedirs(work_dir, exist_ok=True)

    import uuid
    # Append random hex to base to guarantee thread safety for identical uploads
    base       = f"{pathlib.Path(file_path).stem}_{uuid.uuid4().hex[:6]}"
    mp3_path   = os.path.join(work_dir, f"{base}.mp3")
    txt_path   = os.path.join(work_dir, f"{base}.txt")

    # ── Find ffmpeg (checks PATH, winget install, and project-local) ─────────
    def _find_ffmpeg():
        import shutil
        if shutil.which("ffmpeg"):
            return "ffmpeg"
        candidates = [
            os.path.join(REPO_ROOT, "test_video", "ffmpeg.exe"),
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        ]
        # Also scan winget package folder (the exact path varies by version)
        winget_base = os.path.join(os.path.expanduser("~"), "AppData", "Local",
                                   "Microsoft", "WinGet", "Packages")
        if os.path.isdir(winget_base):
            for pkg in os.listdir(winget_base):
                if "Gyan.FFmpeg" in pkg or "ffmpeg" in pkg.lower():
                    for root, _, files in os.walk(os.path.join(winget_base, pkg)):
                        if "ffmpeg.exe" in files:
                            candidates.insert(0, os.path.join(root, "ffmpeg.exe"))
        for p in candidates:
            if os.path.exists(p):
                return p
        return None

    # Step 1 — Convert MP4 → MP3 using ffmpeg
    if not os.path.exists(mp3_path):
        print(f"[Pipeline] 🔄 Extracting audio from video...")
        ffmpeg_cmd = _find_ffmpeg()
        if not ffmpeg_cmd:
            print("[Pipeline] ❌ ffmpeg not found! Install via: winget install Gyan.FFmpeg")
            return 0

        result = subprocess.run(
            [ffmpeg_cmd, "-i", file_path, "-vn", "-acodec", "libmp3lame", "-ab", "192k", mp3_path],
            capture_output=True, text=True
        )
        if not os.path.exists(mp3_path):
            print(f"[Pipeline] ❌ ffmpeg failed: {result.stderr[-500:]}")
            return 0
        print(f"[Pipeline] ✅ Audio extracted → {os.path.basename(mp3_path)}")

    # Step 2 — Whisper transcription
    if not os.path.exists(txt_path):
        print(f"[Pipeline] 🎙️  Transcribing with Whisper...")
        # device="auto" automatically uses CUDA if available, else CPU.
        whisper_model = WhisperModel("large-v3", device="auto", compute_type="default")

        # Quick detection pass
        _, info = whisper_model.transcribe(
            mp3_path, beam_size=5, vad_filter=True,
            initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
        )
        detected_lang = info.language if info.language in ["en", "ur"] else "ur"

        # Full precision transcription
        segments, _ = whisper_model.transcribe(
            mp3_path, language=detected_lang, beam_size=10, vad_filter=True,
            initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
        )
        raw_text = " ".join(s.text.strip() for s in segments)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[Pipeline] ✅ Transcript saved → {os.path.basename(txt_path)}")

    # Step 3 — Refine Urdu transcript with Gemini (skip for English)
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    refined_path = txt_path
    if not is_english(raw_content):
        print(f"[Pipeline] ✏️  Refining Urdu transcript with Gemini...")
        try:
            refined_path = refine_single_file(txt_path)
        except Exception as exc:
            print(f"[Pipeline] ⚠️  Refining skipped: {exc}. Using raw transcript.")
            refined_path = txt_path
    else:
        print(f"[Pipeline] ℹ️  English transcript detected — skipping Gemini refinement.")

    # Step 4 — Translate to create bilingual pair
    print(f"[Pipeline] 🌐 Translating transcript...")
    try:
        translator = TXTDeepTranslator(work_dir)
        translator.process_txt(pathlib.Path(refined_path))
    except Exception as exc:
        print(f"[Pipeline] ⚠️  Translation failed: {exc}")

    # Step 5 — Collect all txt files for this video
    txt_files = []
    for candidate in pathlib.Path(work_dir).glob(f"{base}*.txt"):
        if "_final" not in candidate.stem:
            txt_files.append(str(candidate))
    # Also include the refined txt itself
    if refined_path not in txt_files and os.path.exists(refined_path):
        txt_files.append(refined_path)
    # Deduplicate
    txt_files = list(set(txt_files))

    if not txt_files:
        print(f"[Pipeline] ⚠️  No txt files found to index for this video.")
        return 0

    print(f"[Pipeline] Generated {len(txt_files)} txt file(s): {[os.path.basename(t) for t in txt_files]}")

    # Step 6 — Ingest
    chunks = _ingest_txt_files_to_qdrant(txt_files, category, "video")
    _ingest_txt_files_to_postgres(txt_files, category)

    return chunks


# ==============================================================================
# SECTION 5b — AUDIO SCENARIO
# ==============================================================================

def _run_audio_pipeline(file_path: str, category: str) -> int:
    """
    Processes an uploaded audio file (MP3/WAV/etc.) through:
      1. Faster-Whisper transcription  (audio already on disk — no ffmpeg needed)
      2. Gemini-based Urdu refinement  (skipped for English)
      3. Bilingual translation via TXTDeepTranslator
      4. Qdrant + PostgreSQL ingestion

    Audio resources are invisible in the repository UI; this pipeline exists
    solely to enrich the RAG knowledge base with spoken lecture content.
    Returns total chunk count indexed.
    """
    print(f"\n[Pipeline] 🎧 AUDIO: {os.path.basename(file_path)}")

    from faster_whisper import WhisperModel
    from local_refining import refine_single_file, is_english
    from translating_video import TXTDeepTranslator

    work_dir = os.path.join(AUDIO_WORK_DIR, category)
    os.makedirs(work_dir, exist_ok=True)

    import uuid
    base     = f"{pathlib.Path(file_path).stem}_{uuid.uuid4().hex[:6]}"
    txt_path = os.path.join(work_dir, f"{base}.txt")

    # Step 1 — Whisper transcription directly from the audio file
    # Unlike the video pipeline, we skip ffmpeg because the file is already audio.
    # device="cpu" + compute_type="int8" avoids the cublas64_12.dll CUDA dependency.
    if not os.path.exists(txt_path):
        print(f"[Pipeline] 🎤 Transcribing audio with Whisper...")
        whisper_model = WhisperModel("large-v3", device="cpu", compute_type="int8")

        _, info = whisper_model.transcribe(
            file_path, beam_size=5, vad_filter=True,
            initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
        )
        detected_lang = info.language if info.language in ["en", "ur"] else "ur"

        segments, _ = whisper_model.transcribe(
            file_path, language=detected_lang, beam_size=10, vad_filter=True,
            initial_prompt="PQNK, Emmer Wheat, subsoiler, beds, جنتر، کاشت، کلرٹھی"
        )
        raw_text = " ".join(s.text.strip() for s in segments)

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(raw_text)
        print(f"[Pipeline] ✅ Audio transcript saved → {os.path.basename(txt_path)}")

    # Step 2 — Gemini refinement (Urdu only)
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_content = f.read()

    refined_path = txt_path
    if not is_english(raw_content):
        print(f"[Pipeline] ✏️  Refining Urdu audio transcript with Gemini...")
        try:
            refined_path = refine_single_file(txt_path)
        except Exception as exc:
            print(f"[Pipeline] ⚠️  Refining skipped: {exc}. Using raw transcript.")
            refined_path = txt_path
    else:
        print(f"[Pipeline] ℹ️  English audio detected — skipping Gemini refinement.")

    # Step 3 — Bilingual translation
    print(f"[Pipeline] 🌐 Translating audio transcript...")
    try:
        translator = TXTDeepTranslator(work_dir)
        translator.process_txt(pathlib.Path(refined_path))
    except Exception as exc:
        print(f"[Pipeline] ⚠️  Translation failed: {exc}")

    # Step 4 — Collect txt files
    txt_files = []
    for candidate in pathlib.Path(work_dir).glob(f"{base}*.txt"):
        if "_final" not in candidate.stem:
            txt_files.append(str(candidate))
    if refined_path not in txt_files and os.path.exists(refined_path):
        txt_files.append(refined_path)
    txt_files = list(set(txt_files))

    if not txt_files:
        print(f"[Pipeline] ⚠️  No txt files found to index for this audio.")
        return 0

    print(f"[Pipeline] Generated {len(txt_files)} txt file(s): {[os.path.basename(t) for t in txt_files]}")

    # Step 5 — Ingest to Qdrant + PostgreSQL with source label 'audio'
    chunks = _ingest_txt_files_to_qdrant(txt_files, category, "audio")
    _ingest_txt_files_to_postgres(txt_files, category)

    return chunks


# ==============================================================================
# SECTION 6 — MASTER ORCHESTRATOR
# ==============================================================================

def run_pipeline(file_path: str, file_type: str, category: str, item_id: int = None, **kwargs):
    """
    Entry point called by app.py after a successful upload.
    Runs in a background thread so the HTTP response is not blocked.

    :param file_path:  Absolute local path to the uploaded file.
    :param file_type:  One of 'document', 'image', 'video'.
    :param category:   Category string (e.g. 'soil_science').
    :param item_id:    PostgreSQL Resource.id for status updates (optional).
    """
    fname = os.path.basename(file_path)

    print(f"{'='*60}")
    print(f"[Pipeline] 🚀 Starting pipeline for: {fname}")
    print(f"[Pipeline]    Type: {file_type}  |  Category: {category}")
    print(f"{'='*60}")

    # Step 0 — Sync to Drive (Async)
    # This was moved from app.py to here to make the UI response faster.
    _sync_to_drive_step(file_path, file_type, category, kwargs.get("original_name"), item_id)

    try:
        if file_type == "document":
            total = _run_document_pipeline(file_path, category)

        elif file_type == "image":
            total = _run_image_pipeline(file_path, category)

        elif file_type == "video":
            total = _run_video_pipeline(file_path, category)

        elif file_type == "audio":
            total = _run_audio_pipeline(file_path, category)

        else:
            print(f"[Pipeline] ❌ Unknown file_type: '{file_type}'. Skipping.")
            return

        print(f"\n{'='*60}")
        print(f"[Pipeline] ✅ SUCCESS: '{fname}' → {total} chunks indexed to Qdrant & PostgreSQL")
        print(f"{'='*60}\n")

    except Exception as exc:
        import traceback
        print(f"\n[Pipeline] ❌ FAILED for '{fname}': {exc}")
        traceback.print_exc()


def launch_pipeline_background(file_path: str, file_type: str, category: str, item_id: int = None, original_name: str = None):
    """
    Launches run_pipeline() in a daemon background thread.
    Call this from app.py — it returns immediately.
    """
    t = threading.Thread(
        target=run_pipeline,
        args=(file_path, file_type, category, item_id),
        kwargs={"original_name": original_name},
        daemon=True,
        name=f"pipeline-{os.path.basename(file_path)}"
    )
    t.start()
    print(f"[Pipeline] 🔄 Background task started for '{os.path.basename(file_path)}'")
    return t
