<div align="center">

<img src="../static/logo.png" alt="AgriChat Logo" width="80" style="border-radius:16px" />

# 🌿 PQNK Knowledge Intelligence System

### *AgriChat — AI-Powered Expert Knowledge Repository & Conversational Agent for Sustainable Natural Farming in Pakistan*

[![Stack](https://img.shields.io/badge/Stack-React%20%7C%20Flask%20%7C%20PostgreSQL%20%7C%20Qdrant-2d6a4f?style=flat-square)](https://github.com)
[![AI](https://img.shields.io/badge/AI-RAG%20%7C%20GPT--4o%20%7C%20multimodal-40916c?style=flat-square)](https://openai.com)
[![Language](https://img.shields.io/badge/Languages-English%20%7C%20Urdu-52b788?style=flat-square)](https://github.com)
[![License](https://img.shields.io/badge/License-Academic%20FYP-74c69d?style=flat-square)](https://github.com)

> **Industry Partner:** Pakistan Agriculture Research (PAR) | **University:** Dhanani School of Science & Engineering, Habib University | **Conference:** DURS 2026 — AI for Transforming Agriculture

</div>

---

## 📖 What is PQNK?

**Paedar Qudratti Nizam-e-Kashtari (PQNK)** is a pioneering sustainable natural farming methodology developed by **Dr. Asif Sharif** in collaboration with Pakistan Agriculture Research. It eliminates synthetic chemicals in favour of natural, soil-health-first techniques — a transformative alternative for Pakistan's smallholder farming community.

The problem: **Dr. Sharif's knowledge currently lives in scattered, informal media** — YouTube lectures, WhatsApp broadcasts, documents, social media posts. Farmers only access it through direct, manual consultation with Dr. Sharif himself. This system **cannot scale**.

---

## 🎯 What this System Does

The **PQNK Knowledge Intelligence System** (branded **AgriChat**) digitises and democratises that expert knowledge through two core pillars:

### 1. 🗂️ Multimodal Knowledge Repository
A **searchable, unified repository** of all PQNK content — documents, images, and videos — organised by category, indexed by meaning (not just keywords), and accessible to farmers and researchers through a single web interface.

### 2. 🤖 RAG-Powered Expert Chatbot
A **domain-restricted conversational AI** grounded exclusively in Dr. Sharif's curated corpus. Every response is backed by retrieved source documents — eliminating hallucination risks of general LLMs. Answers are delivered in **Urdu and English with synthesised audio**, lowering literacy barriers for rural users.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          USER LAYER                                  │
│         React 19 + Vite + TailwindCSS  (AgriChat Frontend)          │
│          Farmer / Researcher / Admin / Super Admin                   │
└─────────────────────────┬───────────────────────┬───────────────────┘
                          │                       │
              REST API (Flask)          Auth (Flask-Login + Bcrypt)
                          │                       │
┌─────────────────────────▼───────────────────────▼───────────────────┐
│                       BACKEND LAYER                                  │
│           Flask REST API  ·  Flask-SQLAlchemy  ·  Flask-Bcrypt       │
│           Session Auth  ·  Role-based access (farmer/admin/super)    │
└──────┬────────────────────────┬─────────────────────────────────────┘
       │                        │
       ▼                        ▼
┌──────────────┐    ┌───────────────────────────────────────────────┐
│  PostgreSQL  │    │              INTELLIGENCE LAYER                │
│  User DB     │    │   Qdrant Vector DB  ·  OpenAI GPT-4o          │
│  Auth tables │    │   Sentence Transformers (paraphrase-multilingual)│
│  Sessions    │    │   Edge-TTS (Urdu/English speech synthesis)     │
└──────────────┘    └─────────────────────────┬─────────────────────┘
                                              │
                    ┌─────────────────────────▼─────────────────────┐
                    │             DATA PIPELINE LAYER                │
                    │  PDF extraction → Language detection →         │
                    │  Translation → Chunking → Embedding →          │
                    │  Qdrant ingestion  (scripts in /codes/)        │
                    └───────────────────────────────────────────────┘
```

---

## 🔬 Novelty & Research Contributions

| Aspect | What's Novel |
|--------|-------------|
| **Domain restriction** | Chatbot is RAG-only, grounded 100% in PQNK corpus — no hallucination of general AI knowledge |
| **Multilingual (Urdu + English)** | End-to-end pipeline: Urdu OCR → translation → embedding → Urdu TTS response |
| **Multimodal corpus** | Combines PDFs, images (with GPT-4o vision descriptions), and video transcripts into one searchable knowledge base |
| **Low-literacy access** | Audio responses in Urdu specifically designed for rural, semi-literate farmers |
| **Expert knowledge digitalisation** | First structured digitalisation of the entire PQNK methodology |
| **Scalable template** | Replicable pipeline for any domain expert whose knowledge lives in informal media |

---

## 🛠️ Full Tech Stack

### Frontend
| Technology | Version | Purpose |
|-----------|---------|---------|
| React | 19 | UI framework |
| Vite | 7 | Build tool & dev server |
| TailwindCSS | 3 | Utility-first styling |
| Framer Motion | — | Animations |
| Lucide Icons | — | Icon library |
| React Router DOM | — | Client-side routing |

### Backend
| Technology | Purpose |
|-----------|---------|
| Python 3.13 | Runtime |
| Flask | REST API server |
| Flask-SQLAlchemy | ORM for PostgreSQL |
| Flask-Bcrypt | Password hashing |
| Flask-Login | Session authentication |
| Flask-CORS | Cross-origin support |
| psycopg2 | PostgreSQL adapter |
| python-dotenv | Environment variable management |
| Twilio | SMS OTP delivery |
| smtplib | Email OTP delivery |

### AI / ML
| Technology | Purpose |
|-----------|---------|
| OpenAI GPT-4o | RAG answer generation + image captioning |
| Sentence Transformers (`paraphrase-multilingual-MiniLM-L12-v2`) | Multilingual semantic embeddings |
| Qdrant Cloud | Vector database for semantic search |
| Edge-TTS | Urdu/English text-to-speech synthesis |
| `langdetect` | Query language detection |
| `spellchecker` | Urdu/English query correction |

### Data & Infrastructure
| Technology | Purpose |
|-----------|---------|
| PostgreSQL | User management, auth, roles |
| Google Drive | Primary media storage (docs, images, videos) |
| Qdrant Cloud | Hosted vector index |
| PyMuPDF / pdfminer | PDF text extraction |
| Pillow / OpenCV | Image preprocessing |

---

## 📁 Repository Structure

```
fyp_text/
│
├── about-project/              ← Project documentation (this folder)
│   └── README.md
│
├── agri_ui/                    ← Flask backend
│   ├── app.py                  ← Main Flask app, all REST API endpoints
│   └── rag_demo.py             ← RAG pipeline implementation
│
├── frontend-react/
│   └── PQNK_Frontend/         ← React frontend (Vite + TailwindCSS)
│       ├── src/
│       │   ├── pages/          ← Login, Signup, VerifyOtp, Dashboard, Chatbot, Profile
│       │   ├── layouts/        ← DashboardLayout (sidebar + nav)
│       │   └── routes/         ← AppRoutes.jsx
│       └── public/
│
├── codes/                      ← Data ingestion & processing scripts
│   ├── ingest_data.py          ← Chunk + embed documents into Qdrant
│   ├── migrate_to_qdrant.py    ← FAISS → Qdrant migration
│   ├── extracting_images_from_pdfs.py
│   └── translating_images.py   ← GPT-4o image description pipeline
│
├── .env                        ← 🔒 NOT committed (secrets here)
├── .gitignore
├── abstract_durs.tex           ← DURS 2026 conference abstract (LaTeX)
└── RAG_IMPROVEMENTS.md         ← Documented RAG optimisation roadmap
```

---

## 🚀 Running the Project Locally

### Prerequisites
- Python 3.11+
- Node.js 20+
- PostgreSQL 14+
- Qdrant Cloud account
- OpenAI API key
- Gmail App Password (for email OTP)
- Twilio account (for SMS OTP)

### 1. Clone & setup environment
```bash
git clone https://github.com/your-repo/fyp_text.git
cd fyp_text
cp .env.example .env   # Fill in your credentials
```

### 2. Install Python dependencies
```bash
pip install flask flask-sqlalchemy flask-bcrypt flask-login flask-cors \
            psycopg2-binary python-dotenv openai sentence-transformers \
            qdrant-client langdetect pyspellchecker edge-tts twilio \
            pymupdf pdfminer.six pillow
```

### 3. Set up PostgreSQL
```bash
python setup_database.py
```

### 4. Install & build the frontend
```bash
cd frontend-react/PQNK_Frontend
npm install
npm run build
```

### 5. Start the application
```bash
cd ../../agri_ui
python app.py
```

> The app runs at **http://localhost:5000**. The React frontend dev server runs at **http://localhost:5173** (via `npm run dev`).

---

## 🔒 Environment Variables

Create `.env` in the project root. **Never commit this file.**

```env
# AI & Vector DB
OPENAI_API_KEY=sk-...
QDRANT_API_KEY=...
QDRANT_URL=https://...qdrant.io

# Email OTP (Gmail App Password)
MAIL_USERNAME=your@gmail.com
MAIL_PASSWORD=xxxx xxxx xxxx xxxx

# SMS OTP (Twilio)
TWILIO_ACCOUNT_SID=ACxxxxxxxxxxxxxxxx
TWILIO_AUTH_TOKEN=xxxxxxxxxxxxxxxx
TWILIO_FROM_NUMBER=+15017122661
```

---

## 👥 User Roles

| Role | Access |
|------|--------|
| **Farmer / Seeker** | Chatbot, knowledge repository search, profile |
| **Researcher** | All farmer features + extended repository access |
| **Admin** | Upload & manage content, assign categories/keywords |
| **Super Admin** | Full system access, user role management |

---

## 🌍 Why This Matters

Pakistan's agricultural sector employs **37% of the national workforce** and contributes **~19% of GDP** — yet smallholder farmers, who constitute the vast majority, have almost no access to structured, reliable expert advice in their language.

PQNK's natural farming approach has demonstrated measurable soil health improvements without chemical inputs. The challenge has never been the methodology — it has been **distribution and access**.

AgriChat solves this by converting a single expert's knowledge into a scalable, 24/7 AI system that any farmer in Pakistan can query — in Urdu, from their phone, for free.

---

## 🏆 Awards & Recognition

- Submitted to **DURS 2026** — Dhanani Undergraduate Research Symposium (AI for Agriculture track)
- Industry collaboration with **Pakistan Agriculture Research (PAR)**
- Eligible for **Prototypes for Humanity Dubai 2026** (selected projects share $100,000 award fund)

---

## 📄 Citation

If you reference this work:

```bibtex
@article{pqnk2026,
  title   = {PQNK Knowledge Intelligence System: A RAG-Powered Expert Chatbot
             for Sustainable Natural Farming in Pakistan},
  author  = {[Author Names]},
  journal = {DURS 2026 — AI for Transforming Agriculture},
  year    = {2026},
  institution = {Habib University, Dhanani School of Science \& Engineering}
}
```

---

<div align="center">
  <sub>Built with ❤️ at Habib University · In partnership with Pakistan Agriculture Research</sub>
</div>
