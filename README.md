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

2. **Domain-Restricted RAG Chatbot** — Grounds every response exclusively within Dr. Sharif's curated, approved corpus, eliminating hallucination risks of general-purpose LLMs. Responses are delivered in both **text and synthesised audio** in Urdu and English.

3. **Google Drive Repository** — All uploaded documents, images, and videos are simultaneously stored on Google Drive, organised by type and category. Users can browse and access all resources directly through the website.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│    Login · Signup · Dashboard · Chatbot · Repository · Admin │
└───────────────────────┬──────────────────────────────────────┘
                        │  HTTP (Vite Proxy → :5000)
┌───────────────────────▼──────────────────────────────────────┐
│                     Flask REST API                           │
│   /api/login · /api/signup · /get_response · /generate_audio │
│         /api/repository · /api/admin/*                       │
│          Flask-Login · Flask-CORS · Flask-Bcrypt             │
└──────┬────────────────┬───────────────────────┬──────────────┘
       │                │                       │              │
  ┌────▼────┐    ┌──────▼──────┐         ┌──────▼──────┐ ┌───▼──────────┐
  │PostgreSQL│    │ Qdrant Cloud│         │ OpenAI API  │ │ Google Drive │
  │ (Users & │    │ (Vectors)   │         │ GPT-4o /    │ │  Repository  │
  │Resources)│    │ pqnk_v2     │         │ Embeddings  │ │(PDFs/Images/ │
  └──────────┘    └─────────────┘         └─────────────┘ │   Videos)    │
                                                           └──────────────┘
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
│   ├── app.py                      # Flask REST API (auth, chatbot, TTS, repository)
│   ├── rag_demo.py                 # RAG pipeline (retrieval + generation)
│   ├── drive_service.py            # Google Drive API integration
│   └── templates/                  # Legacy HTML frontend (fallback)
│
├── frontend-react/                 # React Frontend
│   └── PQNK_Frontend/
│       ├── src/
│       │   ├── pages/              # Login, Signup, PublicHome
│       │   │   └── app/            # Dashboard, Chatbot, Admin, SuperAdmin
│       │   ├── components/         # UI components
│       │   ├── layouts/            # Dashboard, Admin, SuperAdmin layouts
│       │   └── routes/             # AppRoutes.jsx
│       └── package.json
│
├── codes/                          # Data processing utilities
│   ├── convert_pdf_to_text.py
│   ├── convert_to_urdu.py
│   ├── convert_urdu_pdf_to_urdu_txt.py
│   ├── convert_urdu_to_eng.py
│   ├── make_embeddings.py
│   ├── merge_embeddings.py
│   └── retrieve.py
│
├── scripts/                        # Utility & maintenance scripts
│   ├── connect_qdrant.py           # Qdrant connectivity test
│   ├── debug_resources.py          # Inspect DB resource records
│   ├── diag_drive.py               # Drive folder structure diagnostic
│   ├── fix_emojis.py               # Fix emoji encoding issues
│   ├── migrate_add_superadmin_role.py
│   ├── migrate_drive_columns.py
│   ├── migrate_to_qdrant.py
│   ├── qdrant_test.py
│   └── test_logic.py
│
├── about-project/                  # Project documentation
│   ├── DEPLOYMENT.md               # Deployment guide
│   └── abstract_durs.tex           # DURS symposium abstract
│
├── ingest_data.py                  # Data ingestion → Qdrant pipeline
├── setup_database.py               # PostgreSQL schema setup
├── seed_users.py                   # Seed initial user accounts
├── generate_oauth_token.py         # One-time Google OAuth2 token generator
├── extracting_images_from_pdfs.py  # Image extraction from PDFs
├── translating_images.py           # Image OCR + translation
├── detection_of_lang_and_renaming.py
│
├── .env                            # ⚠️ NOT committed — contains API keys
├── .env.example                    # Template for environment variables
├── token.json                      # ⚠️ NOT committed — Google OAuth token
└── README.md                       # This file
```

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.10+** with pip
- **Node.js 18+** with npm
- **PostgreSQL 14+** running locally
- **Qdrant Cloud** account
- **OpenAI API** key with GPT-4o access
- **Google Cloud** project with Drive API enabled

### 1. Clone & Setup

```bash
git clone <repository-url>
cd fyp_text
```

### 2. Configure Environment Variables

Copy the example file and fill in your values:

```bash
cp .env.example .env
```

Required variables in `.env`:

```env
# Database
DATABASE_URL=postgresql://postgres:<password>@localhost/pqnk_db

# AI Services
OPENAI_API_KEY=your_openai_api_key
QDRANT_API_KEY=your_qdrant_api_key
QDRANT_URL=https://your-cluster.cloud.qdrant.io

# Google Drive
GOOGLE_DRIVE_CREDENTIALS_JSON=google_service_account.json
GOOGLE_DRIVE_FOLDER_ID=your_root_folder_id

# Flask
SECRET_KEY=your_secret_key
```

### 3. Setup PostgreSQL Database

```bash
psql -U postgres -c "CREATE DATABASE pqnk_db;"
python setup_database.py
python seed_users.py   # Optional: seed test users
```

### 4. Install Python Dependencies

```bash
pip install flask flask-sqlalchemy flask-bcrypt flask-login flask-cors
pip install psycopg2-binary openai qdrant-client
pip install python-dotenv edge-tts pyspellchecker rich
pip install langdetect google-auth google-auth-oauthlib google-api-python-client
```

### 5. Setup Google Drive Integration

1. Go to [Google Cloud Console](https://console.cloud.google.com/) → Enable **Google Drive API**
2. Create an **OAuth Client ID** (Desktop App) → Download as `oauth_credentials.json`
3. Add your Gmail as a **Test User** in the OAuth Consent Screen
4. Run the one-time authorization:
   ```bash
   python generate_oauth_token.py
   ```
   A browser window opens — log in and allow access. This creates `token.json`.

5. Create a `PQNK_LIVE_REPOSITORY` folder on your Drive with subfolders: `PDFs/`, `Images/`, `Videos/`
6. Add the folder ID to your `.env` as `GOOGLE_DRIVE_FOLDER_ID`

> ⚠️ **Never commit `token.json` or `oauth_credentials.json`** — they are already in `.gitignore`

### 6. Install React Frontend

```bash
cd frontend-react/PQNK_Frontend
npm install
```

### 7. Run the Application

**Terminal 1 — Flask Backend:**
```bash
python agri_ui/app.py
# Starts on http://127.0.0.1:5000
```

**Terminal 2 — React Frontend:**
```bash
cd frontend-react/PQNK_Frontend
npm run dev
# Starts on http://localhost:5173
```

Open **http://localhost:5173** in your browser.

---

## 🔌 API Endpoints

### Authentication

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/login` | Login → returns user object |
| `POST` | `/api/signup` | Create new account |
| `GET` | `/api/user` | Get current user |
| `POST` | `/api/logout` | Logout |

### Chatbot

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/get_response` | Send message → AI response |
| `POST` | `/generate_audio` | Text to speech (Urdu/English) |

### Repository

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/repository` | List all resources |
| `POST` | `/api/repository/upload` | Upload file (admin only) → syncs to Drive |
| `DELETE` | `/api/repository/<id>` | Delete resource (admin only) |

### Admin

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/api/admin/stats` | System statistics |
| `GET` | `/api/admin/users` | List all users |
| `PUT` | `/api/admin/users/<id>` | Update user |
| `DELETE` | `/api/admin/users/<id>` | Delete user |

---

## 🗄️ Google Drive Structure

Uploaded resources are automatically organized in Google Drive:

```
PQNK_LIVE_REPOSITORY/
├── PDFs/
│   ├── soil_science/
│   ├── crop_production/
│   └── water_management/ ...
├── Images/
│   ├── PQNK_research_and_knowledge_papers/
│   └── sustainable_agriculture/ ...
└── Videos/
    ├── community_and_farmer_insights/
    └── crop_production/ ...
```

Files are accessible to all users via shareable Drive links shown in the repository browser.

---

## 🌐 Frontend Pages

| Page | Route | Description |
|------|-------|-------------|
| Public Home | `/` | Landing page |
| Login | `/login` | User authentication |
| Signup | `/signup` | Account creation with role selection |
| Dashboard | `/dashboard` | Stats, quick actions, crop advisory |
| Chatbot | `/chatbot` | AI assistant with audio playback |
| Repository | `/content-management` | Browse/upload/delete resources |
| Admin Dashboard | `/admin-dashboard` | User & system management |
| Super Admin | `/super-admin-dashboard` | Full system administration |

---

## 🔑 Key Technologies

| Category | Technology |
|----------|-----------|
| **Frontend** | React 19, Vite 7, TailwindCSS, Framer Motion, Lucide Icons |
| **Backend** | Flask, Flask-Login, Flask-CORS, Flask-Bcrypt, Flask-SQLAlchemy |
| **Database** | PostgreSQL (users & resources), Qdrant Cloud (vectors) |
| **AI/ML** | OpenAI GPT-4o, text-embedding-3-small, Edge TTS |
| **Storage** | Google Drive API (OAuth2) |
| **NLP** | Language detection, spell correction, query expansion |

---

## ⚠️ Security Notes

The following files **must never be committed** to version control (all in `.gitignore`):

| File | Contains |
|------|---------|
| `.env` | All API keys and database credentials |
| `token.json` | Google OAuth2 access/refresh token |
| `oauth_credentials.json` | Google OAuth client secret |
| `google_service_account.json` | Google service account key |

---

## 🌾 Keywords

Retrieval-Augmented Generation (RAG) · Expert Knowledge Systems · Agricultural AI · Sustainable Farming · Natural Language Processing · Multimodal Knowledge Base · Large Language Models · Urdu NLP · Pakistan Agriculture · Google Drive Integration

---

## 📄 License

This project is developed for academic purposes as part of a Final Year Project at Habib University, Karachi, Pakistan.

**Industry Partner:** Pakistan Agriculture Research (PAR)

---

<p align="center">
  <em>Empowering Pakistan's Agriculture with AI 🌿</em>
</p>
