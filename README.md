# PQNK – Precision Agriculture Knowledge Network (AgriChat)

A bilingual (Urdu/English) AI-powered agricultural knowledge platform for Pakistani farmers. Built with a React frontend and a Python/Flask backend using Qdrant vector search + OpenAI GPT-4o.

---

## Architecture Overview

```
frontend-react/PQNK_Frontend/   ← React + Vite + Tailwind (Port 5173)
agri_ui/                        ← Flask backend (Port 5000)
  app.py                        ← Main API server
  rag_demo.py                   ← RAG pipeline (Qdrant + GPT-4o)
  drive_service.py              ← Google Drive sync
uploads/                        ← Local file uploads
static/audio/                   ← TTS audio cache
.env                            ← All secrets and config
```

---

## Prerequisites

| Requirement | Version | Notes |
|---|---|---|
| Python | 3.10 – 3.12 | 3.13 may have compatibility issues |
| Node.js | 18.x or 20.x LTS | — |
| npm | 9+ | Comes with Node |
| PostgreSQL | 14+ | Must be running locally |
| Git | Any | — |

**Cloud services required:**
- [OpenAI API key](https://platform.openai.com/api-keys) (GPT-4o / Whisper)
- [Qdrant Cloud](https://qdrant.tech) (free tier works) — for vector search
- Gmail account with [App Password](https://myaccount.google.com/apppasswords) — for OTP emails
- [Google Cloud Service Account](https://console.cloud.google.com) with Drive API enabled

---

## Step-by-Step Local Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd fyp_text
```

### 2. Set Up the Python Backend

#### 2a. Create a virtual environment

```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

#### 2b. Install Python dependencies

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing, install manually:

```bash
pip install flask flask-sqlalchemy flask-bcrypt flask-login flask-cors
pip install psycopg2-binary sqlalchemy
pip install openai qdrant-client sentence-transformers
pip install python-dotenv langdetect pyspellchecker
pip install twilio requests edge-tts
pip install google-auth google-auth-oauthlib google-api-python-client
pip install werkzeug gunicorn
```

### 3. Configure PostgreSQL

```bash
# Log in to psql
psql -U postgres

# Create the database
CREATE DATABASE pqnk_db;
\q
```

Then run the setup script:

```bash
cd fyp_text
python setup_database.py
```

### 4. Configure Environment Variables

Copy the sample or edit `.env` directly — it already exists in the repo root:

```
fyp_text/.env
```

Fill in the following values:

```env
# ── OpenAI ─────────────────────────────────────────────────────────────────────
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxxxxxxxxxxxxx

# ── Qdrant (Vector Database) ────────────────────────────────────────────────────
QDRANT_URL=https://<your-cluster>.qdrant.io
QDRANT_API_KEY=<your-qdrant-api-key>

# ── PostgreSQL ──────────────────────────────────────────────────────────────────
DB_USER=postgres
DB_PASS=admin123
DB_HOST=localhost
DB_NAME=pqnk_db
SQLALCHEMY_DATABASE_URI=postgresql://postgres:admin123@localhost:5432/pqnk_db

# ── Flask Secret ────────────────────────────────────────────────────────────────
SECRET_KEY=change-this-to-a-random-string-in-production

# ── Email OTP (Gmail App Password) ──────────────────────────────────────────────
# Get App Password from: https://myaccount.google.com/apppasswords
MAIL_USERNAME=your_gmail@gmail.com
MAIL_PASSWORD=xxxx xxxx xxxx xxxx     # 16-character app password
MAIL_FROM_NAME=AgriChat PQNK

# ── SMS OTP via MSG91 (preferred for Pakistan) ───────────────────────────────────
# Sign up at: https://msg91.com/signup
# Get API key from Dashboard → Settings → API Keys
MSG91_API_KEY=your_msg91_api_key_here
MSG91_SENDER_ID=AGRICH
MSG91_TEMPLATE_ID=                    # Optional

# ── SMS OTP via Twilio (fallback) ───────────────────────────────────────────────
# Only needed if MSG91 is not configured
TWILIO_ACCOUNT_SID=your_account_sid_here
TWILIO_AUTH_TOKEN=your_auth_token_here
TWILIO_FROM_NUMBER=+15017122661

# ── Google Drive Integration ─────────────────────────────────────────────────────
# Service account JSON file path
GOOGLE_DRIVE_CREDENTIALS_JSON=C:\path\to\google_service_account.json
GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\google_service_account.json
# The Google Drive folder ID (from the URL: drive.google.com/drive/folders/THIS_ID)
GOOGLE_DRIVE_FOLDER_ID=your_folder_id_here

# ── Google Gemini (optional, for future features) ───────────────────────────────
GEMINI_API_KEY=your_gemini_key_here
```

### 5. Set Up Google Drive Service Account

1. Go to [Google Cloud Console](https://console.cloud.google.com)
2. Create a project → Enable **Google Drive API**
3. Go to **IAM & Admin → Service Accounts → Create**
4. Download the JSON key → save as `google_service_account.json` in `fyp_text/`
5. Open your Google Drive folder → **Share** with the service account email (Editor)
6. Copy the folder ID from the URL and set `GOOGLE_DRIVE_FOLDER_ID` in `.env`

### 6. Set Up the React Frontend

```bash
cd frontend-react/PQNK_Frontend
npm install
```

### 7. Run the Application

**Terminal 1 — Start Flask backend:**
```bash
# From fyp_text/ (with venv activated)
cd agri_ui
python app.py
```
Backend runs at: `http://localhost:5000`

**Terminal 2 — Start React frontend:**
```bash
# From frontend-react/PQNK_Frontend/
npm run dev
```
Frontend runs at: `http://localhost:5173`

Open your browser at **http://localhost:5173**

---

## Setting Up OTP Email (Gmail)

1. Enable **2-Step Verification** on your Gmail account
2. Go to [myaccount.google.com/apppasswords](https://myaccount.google.com/apppasswords)
3. Select "Mail" → "Windows Computer" → Generate
4. Copy the 16-character password into `.env` → `MAIL_PASSWORD`

---

## Setting Up SMS OTP via MSG91 (for Pakistan)

1. Sign up at [msg91.com/signup](https://msg91.com/signup) (free trial credits available)
2. Verify your account and add balance (approx. PKR 2–5 per SMS)
3. Go to **Dashboard → Settings → API Keys** → copy your key
4. Go to **SMS → Sender IDs** → request a 6-char sender ID (e.g. `AGRICH`)
5. Optionally register an SMS template (required for some networks in Pakistan)
6. Set `MSG91_API_KEY`, `MSG91_SENDER_ID`, and optionally `MSG91_TEMPLATE_ID` in `.env`

> **Note:** If MSG91 is not configured, OTP is sent via email only (email works fine by default).

---

## Setting Up Qdrant (Vector DB)

1. Sign up at [cloud.qdrant.io](https://cloud.qdrant.io) (free tier: 1GB)
2. Create a cluster → copy the **URL** and **API Key**
3. Set `QDRANT_URL` and `QDRANT_API_KEY` in `.env`
4. Run the ingestion script to populate vectors:
   ```bash
   python ingest_data.py
   ```

---

## Seeding Initial Admin User

```bash
python seed_users.py
```

Default super admin credentials:
- Email: `admin@pqnk.pk`
- Password: (check `seed_users.py`)

---

## Full Python Dependency List

```
flask>=3.0
flask-sqlalchemy>=3.1
flask-bcrypt>=1.0
flask-login>=0.6
flask-cors>=4.0
psycopg2-binary>=2.9
sqlalchemy>=2.0
openai>=1.30
qdrant-client>=1.9
sentence-transformers>=2.6
python-dotenv>=1.0
langdetect>=1.0.9
pyspellchecker>=0.8
twilio>=8.0
requests>=2.31
edge-tts>=6.1
google-auth>=2.29
google-auth-oauthlib>=1.2
google-api-python-client>=2.127
werkzeug>=3.0
gunicorn>=22.0
```

## Full Frontend Dependency List (from package.json)

```
react, react-dom, react-router-dom
vite (dev)
tailwindcss, autoprefixer, postcss (dev)
```

---

## Common Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Activate venv: `.\venv\Scripts\Activate.ps1` |
| PostgreSQL connection refused | Ensure PostgreSQL service is running: `Start-Service postgresql*` |
| OTP not received | Check `MAIL_USERNAME` and `MAIL_PASSWORD` in `.env`; check spam folder |
| Chatbot returns no answer | Verify Qdrant is populated (`python ingest_data.py`) |
| Google Drive sync fails | Check `GOOGLE_DRIVE_CREDENTIALS_JSON` path and service account permissions |
| Frontend shows CORS error | Ensure Flask runs on port 5000 and `http://localhost:5173` is in CORS origins |
| `SECRET_KEY not set` warning | Set `SECRET_KEY` in `.env` to any random string |

---

## Project Structure

```
fyp_pqnk/
├── agri_ui/               ← Flask backend
│   ├── app.py             ← Main server (routes, auth, repository, chat)
│   ├── rag_demo.py        ← RAG pipeline (retrieval + GPT generation)
│   ├── drive_service.py   ← Google Drive integration
│   └── pipeline_service.py
├── frontend-react/
│   └── PQNK_Frontend/
│       ├── src/
│       │   ├── pages/     ← React pages (Login, Signup, Chatbot, Profile, etc.)
│       │   └── context/   ← Language context (Urdu/English)
│       └── package.json
├── scripts/               ← DB migration and debug scripts
├── uploads/               ← User-uploaded files
├── static/audio/          ← TTS audio cache
├── .env                   ← All configuration (fill before running)
├── setup_database.py      ← DB table creation
├── seed_users.py          ← Creates initial admin user
└── ingest_data.py         ← Ingests documents into Qdrant
```

---

*Built as a Final Year Project (FYP) — PQNK Agricultural Knowledge System, 2026.*
