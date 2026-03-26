<p align="center">
  <img src="https://img.shields.io/badge/Status-In%20Development-green" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/React-19-61DAFB" alt="React">
  <img src="https://img.shields.io/badge/License-Academic-lightgrey" alt="License">
</p>

# 🌿 PQNK Knowledge Intelligence System

### A RAG-Powered Expert Chatbot for Sustainable Natural Farming in Pakistan

> **Final Year Project** — Dhanani School of Science & Engineering, Habib University  
> *Dhanani School Undergraduate Research Symposium (DURS) 2026*  
> Stream: Artificial Intelligence & Machine Learning · Sub-stream: AI for Transforming Agriculture

---

## 📖 Abstract

In Pakistan, expert agronomic knowledge critical to smallholder farmers remains fragmented across informal, heterogeneous media channels — including video lectures, WhatsApp broadcasts, social media posts, and printed documents — rendering it undiscoverable and inaccessible at scale.

This research addresses that gap through the **PQNK Knowledge Intelligence System**, an AI-powered knowledge repository and expert conversational agent developed in collaboration with **Pakistan Agriculture Research (PAR)**.

The system is centred on **Paedar Qudratti Nizam-e-Kashtari (PQNK)**, a sustainable natural farming methodology pioneered by **Dr. Asif Sharif**, whose domain expertise currently reaches farmers exclusively through direct, manual consultation. Our goal is to **automate and scale** that consultation process without compromising the integrity of the source knowledge.

### Core Contributions

1. **Multimodal Knowledge Base** — Systematically collected and processed the scattered PQNK corpus. Raw content — spanning multiple languages and formats — undergoes language detection, machine translation, semantic chunking, and vector embedding, indexed into a Qdrant vector store for efficient retrieval.

2. **Domain-Restricted RAG Chatbot** — Grounds every response exclusively within Dr. Sharif's curated, approved corpus, eliminating hallucination risks of general-purpose LLMs. Responses are delivered in both **text and synthesised audio** in Urdu and English, deliberately lowering literacy and accessibility barriers for rural users.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│          Login · Signup · Dashboard · Chatbot · Admin        │
│              TailwindCSS · Framer Motion · Lucide            │
└───────────────────────┬──────────────────────────────────────┘
                        │  HTTP (Vite Proxy → :5000)
┌───────────────────────▼──────────────────────────────────────┐
│                     Flask REST API                           │
│   /api/login · /api/signup · /get_response · /generate_audio │
│          Flask-Login · Flask-CORS · Flask-Bcrypt             │
└──────┬────────────────┬────────────────────────┬─────────────┘
       │                │                        │
  ┌────▼────┐    ┌──────▼──────┐          ┌──────▼──────┐
  │PostgreSQL│    │ Qdrant Cloud│          │ OpenAI API  │
  │ (Users)  │    │ (Vectors)   │          │ GPT-4o /    │
  │          │    │ pqnk_v2     │          │ Embeddings  │
  └──────────┘    └─────────────┘          └─────────────┘
```

---

## 🧠 RAG Pipeline Overview

The core intelligence resides in `agri_ui/rag_demo.py`:

| Stage | Description |
|-------|-------------|
| **Query Preprocessing** | Normalisation → Spell correction → Entity/acronym resolution → Intent normalisation |
| **Language Detection** | Custom Urdu/English detector via character-ratio analysis |
| **Hybrid Retrieval** | Original query retrieval + Urdu→English translation retrieval, merged and deduplicated |
| **Multi-hop Retrieval** | For complex queries (why/how/impact), performs two-pass retrieval with context enrichment |
| **Query Expansion** | Paraphrasing (GPT-4o-mini) + agricultural domain expansion + intent compression |
| **Reranking** | Score-based ranking with language-match boosting and minimum score threshold filtering |
| **Answer Generation** | GPT-4o with domain-specific system prompts, language-appropriate formatting |
| **Text-to-Speech** | Edge TTS for Urdu (`ur-PK-UzmaNeural`) and English (`en-US-AriaNeural`) audio output |

### Key Models Used

| Purpose | Model | Provider |
|---------|-------|----------|
| Embeddings | `text-embedding-3-small` | OpenAI |
| Generation | `gpt-4o` | OpenAI |
| Query Intelligence | `gpt-4o-mini` | OpenAI |
| Text-to-Speech | Edge TTS | Microsoft |
| Vector Storage | Qdrant Cloud | Qdrant |

---

## 📂 Project Structure

```
fyp_text/
│
├── agri_ui/                        # Backend application
│   ├── app.py                      # Flask REST API (auth, chatbot, TTS, JSON API)
│   ├── rag_demo.py                 # RAG pipeline (retrieval + generation)
│   ├── templates/                  # Legacy HTML/CSS frontend (fallback)
│   │   ├── login.html
│   │   ├── signup.html
│   │   ├── dashboard.html
│   │   └── chatbot.html
│   └── static/                     # Static assets for legacy frontend
│
├── frontend-react/                 # Official React Frontend
│   └── PQNK_Frontend/
│       ├── src/
│       │   ├── pages/              # Login, Signup, PublicHome
│       │   │   └── app/            # Dashboard, Chatbot, Admin, SuperAdmin
│       │   ├── components/         # UI components (layout, ui primitives)
│       │   ├── layouts/            # Dashboard, Admin, SuperAdmin layouts
│       │   ├── routes/             # AppRoutes.jsx (routing config)
│       │   └── main.jsx            # Entry point
│       ├── vite.config.js          # Vite config with Flask proxy
│       ├── tailwind.config.js      # TailwindCSS configuration
│       └── package.json            # Dependencies
│
├── codes/                          # Data processing utilities
│   ├── convert_pdf_to_text.py      # PDF → Plain text extraction
│   ├── convert_to_urdu.py          # English → Urdu translation
│   ├── convert_urdu_pdf_to_urdu_txt.py  # Urdu PDF → text
│   ├── convert_urdu_to_eng.py      # Urdu → English translation
│   ├── database.py                 # PostgreSQL database utilities
│   ├── make_embeddings.py          # Generate vector embeddings
│   ├── merge_embeddings.py         # Merge embedding datasets
│   └── retrieve.py                 # Retrieval testing utilities
│
├── text_pdfs/                      # Source PQNK documents (368 files)
├── images/                         # Extracted PDF images (432 files)
├── embeddings_output/              # Pre-computed embeddings
├── faiss_indexes/                  # Legacy FAISS indexes (deprecated)
├── static/                         # Global static assets (audio, images, CSS)
│
├── ingest_data.py                  # Main data ingestion → Qdrant pipeline
├── migrate_to_qdrant.py            # Migration from local → Qdrant Cloud
├── setup_database.py               # PostgreSQL schema setup
├── seed_users.py                   # Seed initial user accounts
├── extracting_images_from_pdfs.py  # Image extraction from PDFs
├── translating_images.py           # Image OCR + translation
├── detection_of_lang_and_renaming.py  # Language detection for files
│
├── .env                            # Environment variables (API keys)
├── abstract_durs.tex               # DURS research symposium abstract
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **PostgreSQL 14+** running locally
- **Qdrant Cloud** account (or local Qdrant instance)
- **OpenAI API** key with GPT-4o access

### 1. Clone & Setup Environment

```bash
git clone <repository-url>
cd fyp_text
```

### 2. Configure Environment Variables

Create a `.env` file in the project root:

```env
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io
OPENAI_API_KEY=your_openai_api_key
```

### 3. Setup PostgreSQL Database

```bash
# Create the database
psql -U postgres -c "CREATE DATABASE pqnk_db;"

# Setup tables
python setup_database.py

# (Optional) Seed test users
python seed_users.py
```

### 4. Install Python Dependencies

```bash
pip install flask flask-sqlalchemy flask-bcrypt flask-login flask-cors
pip install psycopg2-binary openai qdrant-client
pip install python-dotenv edge-tts pyspellchecker rich
pip install langdetect sentence-transformers
```

### 5. Install React Frontend Dependencies

```bash
cd frontend-react/PQNK_Frontend
npm install
```

### 6. Run the Application

You need **two terminal windows**:

**Terminal 1 — Flask Backend:**
```bash
cd agri_ui
python app.py
# Backend starts on http://127.0.0.1:5000
```

**Terminal 2 — React Frontend:**
```bash
cd frontend-react/PQNK_Frontend
npm run dev
# Frontend starts on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## 🔌 API Endpoints

### Authentication (JSON API)

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/login` | Login with email/password → returns user object |
| `POST` | `/api/signup` | Create new account |
| `GET` | `/api/user` | Get current authenticated user |
| `POST` | `/api/logout` | Logout current session |

### Chatbot

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/get_response` | Send message to RAG pipeline → returns AI response |
| `POST` | `/generate_audio` | Convert text to speech (Urdu/English) → returns audio URL |

### Legacy HTML Routes

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET/POST` | `/` | HTML login form |
| `GET/POST` | `/signup` | HTML signup form |
| `GET` | `/dashboard` | HTML dashboard |
| `GET` | `/chatbot` | HTML chatbot interface |

---

## 📊 Data Pipeline

The knowledge base was constructed through the following pipeline:

```
Raw PDFs (Urdu + English)
    │
    ├── extracting_images_from_pdfs.py  → images/
    ├── codes/convert_pdf_to_text.py    → text_pdfs/
    ├── detection_of_lang_and_renaming.py
    │
    ├── codes/convert_urdu_to_eng.py    → Translated texts
    ├── codes/convert_to_urdu.py        → Urdu translations
    │
    ├── codes/make_embeddings.py        → embeddings_output/
    ├── codes/merge_embeddings.py       → Merged datasets
    │
    └── ingest_data.py                  → Qdrant Cloud (pqnk_v2)
```

**Vector Collection:** `pqnk_v2` on Qdrant Cloud  
**Embedding Model:** `text-embedding-3-small` (1536 dimensions)  
**Chunking:** 1000 chars with 200 overlap, language-aware separators

---

## 🌐 Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Public Home | `/` | Landing page |
| Login | `/login` | User authentication |
| Signup | `/signup` | New account creation with role selection |
| Dashboard | `/dashboard` | Stats, quick actions, crop advisory, soil health |
| Chatbot | `/chatbot` | AI assistant with audio playback, copy, feedback |
| Admin Dashboard | `/admin-dashboard` | Admin management panel |
| Super Admin | `/super-admin-dashboard` | System-wide administration |

---

## 🔑 Key Technologies

| Category | Technology |
|----------|-----------|
| **Frontend** | React 19, Vite 7, TailwindCSS 3, Framer Motion, Lucide Icons |
| **Backend** | Flask, Flask-Login, Flask-CORS, Flask-Bcrypt, Flask-SQLAlchemy |
| **Database** | PostgreSQL (users), Qdrant Cloud (vectors) |
| **AI/ML** | OpenAI GPT-4o, text-embedding-3-small, Edge TTS |
| **NLP** | Language detection, spell correction, query expansion, intent compression |
| **Data** | LangChain text splitters, PyMuPDF, langdetect |

---

## 🌾 Keywords

Retrieval-Augmented Generation (RAG) · Expert Knowledge Systems · Agricultural AI · Sustainable Farming · Natural Language Processing · Multimodal Knowledge Base · Large Language Models · Urdu NLP · Pakistan Agriculture

---

## 📄 License

This project is developed for academic purposes as part of a Final Year Project at Habib University, Karachi, Pakistan.

**Industry Partner:** Pakistan Agriculture Research (PAR)

---

<p align="center">
  <em>Empowering Pakistan's Agriculture with AI 🌿</em>
</p>
