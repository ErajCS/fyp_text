# PQNK Automation Pipeline — Architecture & Implementation Plan

> **Purpose:** Complete technical specification for the automated content ingestion pipeline.
> When an admin uploads a document, image, or video link via the dashboard, this pipeline fires automatically.

---

## Overview

```
Admin uploads via Dashboard
        │
        ▼
[1] Flask receives file / URL
        │
        ▼
[2] Save to Google Drive (via Drive API)
        │
        ▼
[3] Detect content type (PDF · Image · Video)
        │
        ▼
[4] Content-specific extraction
    ├── PDF  → text extraction (PyMuPDF)
    ├── Image → GPT-4o vision description
    └── Video → audio extraction → Whisper transcription
        │
        ▼
[5] Language detection + translation (Urdu → English)
        │
        ▼
[6] Semantic chunking
        │
        ▼
[7] Embedding generation (paraphrase-multilingual)
        │
        ▼
[8] Upsert into Qdrant with metadata
    (category, keywords, source, type, drive_id)
        │
        ▼
[9] Save record to PostgreSQL (content_items table)
        │
        ▼
[10] Notify admin → "Uploaded & indexed successfully"
```

---

## Step-by-Step Implementation

### Step 1 & 2 — File Upload + Google Drive Save

**API endpoint:** `POST /api/admin/upload`

The Flask endpoint receives the multipart form upload, then uses the **Google Drive API v3** to upload the file into the correct category folder on Drive.

```python
# Google Drive upload
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from google.oauth2 import service_account

SCOPES = ['https://www.googleapis.com/auth/drive.file']
SERVICE_ACCOUNT_FILE = 'service-account-key.json'  # ← in .gitignore

creds = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES)
drive_service = build('drive', 'v3', credentials=creds)

def upload_to_drive(file_path, filename, category_folder_id):
    file_metadata = {
        'name': filename,
        'parents': [category_folder_id]
    }
    media = MediaFileUpload(file_path, resumable=True)
    uploaded = drive_service.files().create(
        body=file_metadata, media_body=media, fields='id,webViewLink'
    ).execute()
    return uploaded['id'], uploaded['webViewLink']
```

**Setting up Drive API:**
1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Enable **Google Drive API** on your project
3. Create a **Service Account** → download JSON key → add to `.env` (path)
4. Share each Drive category folder with the service account email (Editor)

---

### Step 3 — Content Type Detection

```python
import mimetypes

def detect_content_type(filename: str, url: str = None) -> str:
    if url:
        if 'youtube.com' in url or 'youtu.be' in url:
            return 'video'
        return 'url'
    mime, _ = mimetypes.guess_type(filename)
    if mime:
        if 'pdf' in mime:         return 'pdf'
        if mime.startswith('image'): return 'image'
        if mime.startswith('video'): return 'video'
    ext = filename.rsplit('.', 1)[-1].lower()
    return {'pdf': 'pdf', 'docx': 'docx', 'png': 'image',
            'jpg': 'image', 'mp4': 'video'}.get(ext, 'unknown')
```

---

### Step 4 — Content-Specific Extraction

#### 4a. PDF → Text
```python
import fitz  # PyMuPDF

def extract_pdf_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    return "\n\n".join(page.get_text() for page in doc)
```

#### 4b. Image → GPT-4o Description
```python
import base64, openai

def describe_image(image_path: str) -> str:
    with open(image_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this agricultural image in detail. Include any text visible, farming techniques shown, plants, soil conditions, and any PQNK-related content."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
            ]
        }]
    )
    return response.choices[0].message.content
```

#### 4c. YouTube Video → Transcript
```python
from youtube_transcript_api import YouTubeTranscriptApi
import re

def get_youtube_transcript(youtube_url: str) -> str:
    video_id = re.search(r'(?:v=|youtu\.be/)([^&\n?]+)', youtube_url).group(1)
    try:
        transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['ur', 'en'])
        return " ".join(chunk['text'] for chunk in transcript)
    except Exception:
        # Fallback: download audio → Whisper
        return extract_audio_and_transcribe(youtube_url)

def extract_audio_and_transcribe(url: str) -> str:
    import yt_dlp, openai
    opts = {'format': 'bestaudio', 'outtmpl': '/tmp/audio.%(ext)s', 'quiet': True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    client = openai.OpenAI()
    with open('/tmp/audio.webm', 'rb') as f:
        result = client.audio.transcriptions.create(model='whisper-1', file=f)
    return result.text
```

#### 4d. Local Video → Audio → Whisper
```python
import subprocess, openai

def transcribe_video(video_path: str) -> str:
    audio_path = video_path.replace('.mp4', '_audio.mp3')
    subprocess.run(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', audio_path], check=True)
    client = openai.OpenAI()
    with open(audio_path, 'rb') as f:
        result = client.audio.transcriptions.create(model='whisper-1', file=f)
    return result.text
```

---

### Step 5 — Language Detection + Translation

```python
from langdetect import detect
from deep_translator import GoogleTranslator

def detect_and_translate(text: str) -> tuple[str, str]:
    """Returns (original_lang, english_text)"""
    try:
        lang = detect(text[:500])
    except:
        lang = 'en'
    if lang == 'ur':
        translated = GoogleTranslator(source='ur', target='en').translate(text[:4500])
        return 'ur', translated
    return lang, text
```

---

### Step 6 — Semantic Chunking

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """Splits text into overlapping semantic chunks."""
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        i += chunk_size - overlap
    return [c for c in chunks if len(c.strip()) > 50]  # filter tiny chunks
```

---

### Step 7 — Embedding Generation

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

def generate_embeddings(chunks: list[str]) -> list[list[float]]:
    return model.encode(chunks, convert_to_numpy=True).tolist()
```

---

### Step 8 — Upsert into Qdrant

```python
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct
import uuid

client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
COLLECTION = "pqnk_knowledge"

def upsert_to_qdrant(chunks, embeddings, metadata: dict):
    points = []
    for i, (chunk, vec) in enumerate(zip(chunks, embeddings)):
        points.append(PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload={
                "text": chunk,
                "source": metadata['filename'],
                "category": metadata['category'],
                "keywords": metadata['keywords'],
                "content_type": metadata['content_type'],
                "drive_id": metadata['drive_id'],
                "drive_url": metadata['drive_url'],
                "language": metadata['language'],
                "chunk_index": i,
            }
        ))
    client.upsert(collection_name=COLLECTION, points=points)
    print(f"✅ Upserted {len(points)} chunks to Qdrant")
```

---

### Step 9 — PostgreSQL Record

```python
# New table needed: content_items
class ContentItem(db.Model):
    __tablename__ = 'content_items'
    id           = db.Column(db.Integer, primary_key=True)
    filename     = db.Column(db.String(500), nullable=False)
    content_type = db.Column(db.String(50))   # pdf / image / video / url
    category     = db.Column(db.String(200))
    keywords     = db.Column(db.ARRAY(db.String))
    drive_id     = db.Column(db.String(200))
    drive_url    = db.Column(db.String(500))
    uploaded_by  = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    uploaded_at  = db.Column(db.DateTime, default=datetime.utcnow)
    chunk_count  = db.Column(db.Integer)
    status       = db.Column(db.String(50), default='indexed')  # indexed / failed
```

---

### Step 10 — Master Pipeline Function

```python
def run_ingestion_pipeline(file_path, filename, category, keywords,
                            drive_folder_id, uploader_id, url=None):
    # 1. Upload to Drive
    drive_id, drive_url = upload_to_drive(file_path or url, filename, drive_folder_id)

    # 2. Detect content type
    ctype = detect_content_type(filename, url)

    # 3. Extract content
    if ctype == 'pdf':     raw_text = extract_pdf_text(file_path)
    elif ctype == 'image': raw_text = describe_image(file_path)
    elif ctype == 'video':
        raw_text = get_youtube_transcript(url) if url else transcribe_video(file_path)
    else:
        raw_text = ""

    # 4. Translate if Urdu
    lang, text_en = detect_and_translate(raw_text)

    # 5. Chunk
    chunks = chunk_text(text_en)

    # 6. Embed
    embeddings = generate_embeddings(chunks)

    # 7. Upsert to Qdrant
    meta = dict(filename=filename, category=category, keywords=keywords,
                content_type=ctype, drive_id=drive_id, drive_url=drive_url, language=lang)
    upsert_to_qdrant(chunks, embeddings, meta)

    # 8. Save to PostgreSQL
    item = ContentItem(filename=filename, content_type=ctype, category=category,
                       keywords=keywords, drive_id=drive_id, drive_url=drive_url,
                       uploaded_by=uploader_id, chunk_count=len(chunks))
    db.session.add(item)
    db.session.commit()

    return {"success": True, "chunks": len(chunks), "drive_url": drive_url}
```

---

## New Python Packages Required

```bash
pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
pip install youtube-transcript-api yt-dlp openai-whisper deep-translator
# ffmpeg must be installed as a system tool (not pip)
```

---

## Timeline Estimate

| Phase | Task | Estimated Time |
|-------|------|---------------|
| 1 | Google Drive API setup + service account | 2–3 hrs |
| 2 | Admin upload UI (React) | 4–6 hrs |
| 3 | Flask upload endpoint + Drive save | 3–4 hrs |
| 4 | Content extraction per type | 4–6 hrs |
| 5 | Pipeline integration + testing | 4–6 hrs |
| 6 | Repository search page (React) | 4–6 hrs |
| **Total** | | **~21–31 hrs** |
