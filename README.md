<p align="center">
  <img src="https://img.shields.io/badge/Status-Beta%20Deployment-green" alt="Status">
  <img src="https://img.shields.io/badge/Python-3.13-blue" alt="Python">
  <img src="https://img.shields.io/badge/React-19-61DAFB" alt="React">
  <img src="https://img.shields.io/badge/Database-PostgreSQL%20|%20Qdrant-orange" alt="Data">
</p>

# 🌿 PQNK Knowledge Intelligence System

### An AI-Augmented Platform for Sustainable Natural Farming Advisory

> **Industry Partner:** Pakistan Agriculture Research (PAR) <br/>
> Developed in collaboration with Dhanani School of Science & Engineering, Habib University.

---

## 📖 Executive Summary

The **PQNK Knowledge Intelligence System** is an enterprise-grade web platform and Retrieval-Augmented Generation (RAG) conversational agent. It is designed to digitize, index, and autonomously distribute the expert agronomic methodology of **Paedar Qudratti Nizam-e-Kashtari (PQNK)**, pioneered by **Dr. Asif Sharif**. 

Currently, farmers rely on scattered, heterogeneous media (WhatsApp audio, PDFs, YouTube videos) to receive guidance. This platform unifies that knowledge, providing a centralized repository, an asynchronous AI parsing engine, and an expert-grounded bilingual chatbot that speaks Urdu and English—providing rural farmers with 24/7 localized guidance with zero hallucination risk.

### 🌟 Core Capabilities

1. **Multimodal Ingestion Engine:** Automatically processes Documents (with embedded image OCR), Stand-alone Images, Audio notes, and Video lectures. Runs asynchronously using a background daemon thread.
2. **Domain-Restricted RAG Chatbot:** Provides highly precise answers grounded *strictly* in the approved PQNK knowledge base.
3. **Bilingual Neural Output:** Translates content seamlessly and delivers answers in both formatted text and localized Edge TTS Neural Audio.
4. **Google Drive Cloud Syncing:** Maps database categories to automated Google Drive subfolders, providing immediate cloud backup and clickable provenance links.
5. **Dual-Channel Authentication:** Secure Role-Based Access Control (RBAC) backed by Twilio SMS and Gmail SMTP One-Time Passwords (OTP).

---

## 🏗️ System Architecture & Technology Stack

The platform is decoupled into a robust modern web architecture:

### 1. Frontend Client (React)
- **Framework:** React 19 + Vite 7
- **Styling:** TailwindCSS, Framer Motion (micro-animations), Lucide Icons
- **Key Modules:** 
  - Dynamic interactive Repository Browser.
  - Live Audio-integrated Chatbot UI.
  - Multi-tier Dashboards (Seeker, Farmer, Admin, Super Admin).

### 2. Backend API (Flask)
- **Framework:** Python (Flask 3)
- **Concurrency:** Threading (Daemon background workers) for async data ingestion to prevent UI blocking.
- **Security:** Flask-Bcrypt (hashing), Flask-Login (session management), JWT/Session hybrid.
- **Cloud Connectivity:** Google API Client (OAuth2) for automated Drive synchronization.

### 3. Data & AI Layer
- **Relational Database:** **PostgreSQL 14+** (managed via SQLAlchemy) for users, authentication metrics, and file metadata.
- **Vector Store:** **Qdrant Cloud** to handle dense semantic embeddings (`pqnk_v2` index).
- **Audio to Text:** **Faster-Whisper** (`large-v3`, forced to CPU/int8) for lightning-fast localized Urdu/English audio transcription.
- **OCR & Vision:** **PyMuPDF**, **img2table**, **Tesseract**, and **GPT-4o-mini** for extracting text from scattered embedded images in textbooks.
- **Intelligence Generation:** **OpenAI GPT-4o** (Chat) and `text-embedding-3-small` (Vector Generation).

<br/>

```text
┌──────────────────────────────────────────────────────────────┐
│                     React Frontend (Vite)                    │
│    Login · OTP Auth · Chatbot · Repository · Admin Panel     │
└───────────────────────┬──────────────────────────────────────┘
                        │  REST HTTP
┌───────────────────────▼──────────────────────────────────────┐
│                 Flask API & Async Worker                     │
│               [Multimodal Ingestion Pipeline]                │
└──────┬────────────────┬───────────────────────┬──────────────┘
       │                │                       │              │
  ┌────▼────┐    ┌──────▼──────┐         ┌──────▼──────┐ ┌───▼──────────┐
  │PostgreSQL│   │ Qdrant Cloud│         │ Local CPU   │ │ Google Drive │
  │(Metadata)│   │ (Vectors)   │         │ (Whisper)   │ │ (Raw Files)  │
  └──────────┘   └─────────────┘         └─────────────┘ └──────────────┘
```

---

## ⚙️ The Multimodal Ingestion Pipeline

When an administrator uploads a core asset to the platform, a highly complex **10-stage background pipeline** invokes automatically:

1. **Synchronized Upload:** The file is immediately mapped to the correct structural folder in Google Drive. 
2. **Audio/Video Rip:** `ffmpeg` strips audio from video lectures and passes it to the local `Faster-Whisper` CPU model.
3. **Visual Extraction:** PDFs are deeply scanned. Text is stripped, and embedded graphs/tables are extracted, run through Tesseract OCR, and captioned by GPT-4 Vision.
4. **Bilingual Translation:** Sourced content is translated to create a "Mirror Index", ensuring English queries map to Urdu documents, and vice versa.
5. **Vectorizing & Qdrant Upsert:** Text is chunked, converted to dense arrays via `text-embedding-3-small`, and payload metadata (Google Drive Links, Origin markers) is embedded into the Qdrant Cloud.

---

## 🚀 Deployment & Installation Guide

For industry partners assessing deployment environments, this system can be deployed onto Linux/Windows server environments (AWS EC2, Digital Ocean Droplet) or via Docker.

### Prerequisites
- **Python 3.10+** and **Node.js 18+**
- Running instance of **PostgreSQL**
- Accounts for: **OpenAI**, **Twilio** (SMS Auth), **Qdrant Cloud**, and **Google Cloud Console** (Drive API).

### Step 1: Clone and Env Setup
```bash
git clone <repository-url>
cd fyp_text
cp .env.example .env
```
Ensure database credentials, API keys (`OPENAI_API_KEY`, `TWILIO_SID`, etc.), and Google secrets are populated in `.env`.

### Step 2: Google Drive Service Authentication
You must provide the application permission to act as an OAuth user to provision storage and bypass the 0-byte Service Account Quota.
```bash
python generate_oauth_token.py
```
*(This will generate the locally bound `token.json` used by the sync service).*

### Step 3: Database Migration
```bash
python setup_database.py
python seed_users.py  # Deploys the root Super Admin account
```

### Step 4: Starting the Services

**Terminal 1 — Flask API Engine**
```bash
cd agri_ui
pip install -r requirements.txt # (Dependencies: flask, faster-whisper, openai, sqlalchemy, etc)
python app.py
# Runs production or development WSGI on Port 5000
```

**Terminal 2 — React Client**
```bash
cd frontend-react/PQNK_Frontend
npm install
npm run dev
# Vite runs on Port 5173
```

---

## 🛡️ Security Posture & Compliance

- **Identity Verification:** Dual-layer MFA (Twilio SMS & Google SMTP) mandatory on signup prior to role assignment. 
- **Secret Management:** `.env`, `.pem` keys, and `token.json` are strictly `.gitignore`'d and handled through server orchestration secrets.
- **RBAC:** Routes are protected at the React UI level and enforced strictly on the Flask API decorators to prevent standard users from accessing Admin/Write endpoints.

---

## 📂 Code Repository Map

```text
fyp_text/
├── agri_ui/                        # Backend Application Kernel
│   ├── app.py                      # Core Web API & Auth Logic
│   ├── pipeline_service.py         # Async Worker & Data Extraction Logic
│   ├── drive_service.py            # OAuth & Sync Module
│   └── rag_demo.py                 # Retrieval/Qdrant + OpenAI Interaction
├── frontend-react/                 # React UI Stack
│   └── PQNK_Frontend/src/
│       ├── components/             # Reusable UI forms & loaders 
│       ├── layouts/                # Admin sidebar wrappers
│       └── pages/                  # Views (Repository.jsx, Chatbot.jsx)
├── about-project/                  # Formal documentation, Thesis (.tex)
├── token.json                      # [Ignored] Local Google Auth Bind
└── .env                            # [Ignored] Centralized Environment Configuration
```

---

<p align="center">
  <em>Designed for <b>Pakistan Agriculture Research (PAR)</b>.<br>Bridging the gap between Agronomic Experts and the Pakistani Farmer.</em>
</p>
