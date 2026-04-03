# 🚀 PQNK Knowledge Intelligence System — Comprehensive Deployment & Architecture Guide

This document provides a detailed, step-by-step guide to take the PQNK system from your local development environment to a production reality.

---

## 1. Where to Deploy: Azure vs. "Free" Platforms

**Verdict:** You **MUST** use a Virtual Machine (like Microsoft Azure). "Free" Platforms (like Vercel, Render, Heroku free tiers) will **NOT** work for the backend.

### Why not entirely free platforms?
While your React Frontend can be hosted for free on Vercel, your Flask Backend contains heavy data science pipelines. Specifically:
1. **Faster-Whisper (`large-v3`)**: Requires at least 4GB to 8GB of RAM to transcribe audio without crashing.
2. **PyMuPDF & Tesseract OCR**: Consumes significant CPU and memory when processing large documents.
3. **Background Daemon Threads**: Your ingest pipeline runs asynchronous threading (`threading.Thread`). Serverless environments (like Vercel functions, Heroku basic) freeze threads as soon as the HTTP request returns, which will instantly kill your upload pipelines.

### The Recommended Solution: Azure for Students
As a university student, you get **$100 in free Azure credits** plus free services. 
- You should deploy the **Frontend on Vercel** (Free).
- You should deploy the **Backend + PostgreSQL on an Azure Virtual Machine (VM)** (e.g., `Standard_B2ms` or `Standard_B4ms` running Ubuntu Linux).

---

## 2. How the System Looks Different After Deployment

Currently, when you run locally, you open two terminals, type `python app.py`, and `npm run dev`. In production, the system architecture shifts from a "Debug" state to a "Production" state:

| Component | Local Development (Current) | Production Deployment (Future) |
| :--- | :--- | :--- |
| **Frontend Server** | Vite Dev Server (`localhost:5173`) | Vercel Global Edge CDN (`pqnk.vercel.app`) |
| **Backend Server** | Flask Development Server (`localhost:5000`) | Gunicorn WSGI Server bound to Nginx Reverse Proxy |
| **Network Security** | HTTP | HTTPS (SSL/TLS certificates provided via Certbot/Let's Encrypt) |
| **Database (PostgreSQL)** | Local Windows pgAdmin/Postgres Service | PostgreSQL installed on Linux Azure VM |
| **Continuous Uptime** | Stops when you close the terminal window | Managed by `systemd` (runs forever in the background) |
| **Vector Database** | Qdrant Cloud | Continues using Qdrant Cloud |

### What EXACTLY needs to be deployed?
To make the system live globally, you must independently deploy:
1. **The Database:** Export your local PostgreSQL database schemas/tables and recreate them on the cloud VM.
2. **The Backend (Flask API + Pipelines):** Hosted on the Azure VM, running behind Gunicorn and Nginx. This includes `ffmpeg`, `Tesseract`, and all Python models.
3. **The Frontend (React):** Built and deployed to Vercel, programmed to send API requests to your public Azure VM's IP address/Domain instead of `localhost:5000`.

---

## 3. Step-by-Step Deployment Guide

### Step A: Set up the Azure Virtual Machine (Backend & Database)
1. Go to the [Azure Portal](https://portal.azure.com/) and sign in with your student account.
2. Create a new **Virtual Machine**.
   - **OS:** Ubuntu Server 22.04 LTS (or 24.04).
   - **Size:** Select `Standard_B2ms` (2 vCPUs, 8GB RAM) or similar.
   - **Inbound Ports:** Allow `SSH (22)`, `HTTP (80)`, and `HTTPS (443)`.
3. SSH into your newly created VM from your local terminal:
   ```bash
   ssh azureuser@<YOUR_VM_PUBLIC_IP>
   ```

### Step B: Environment Preparation on the Linux VM
Once inside the VM, install system dependencies:
```bash
sudo apt update
sudo apt install python3-pip python3-venv postgresql postgresql-contrib nginx tesseract-ocr tesseract-ocr-urd ffmpeg -y
```
Notice we are installing `ffmpeg` and `tesseract-ocr` at the OS level, eliminating the need to package the `.exe` files you currently use on Windows.

### Step C: Deploy the PostgreSQL Database
1. Switch to the postgres user and open the console:
   ```bash
   sudo -u postgres psql
   ```
2. Create the PQNK database and user:
   ```sql
   CREATE DATABASE pqnk_db;
   CREATE USER pqnk_user WITH PASSWORD 'your_secure_password';
   GRANT ALL PRIVILEGES ON DATABASE pqnk_db TO pqnk_user;
   ALTER DATABASE pqnk_db OWNER TO pqnk_user;
   \q
   ```
3. Run the schema creation script from your cloned repository on the VM.

### Step D: Deploy the Flask Backend
1. Clone your GitHub repository inside the VM:
   ```bash
   git clone https://github.com/YOUR_GITHUB/pqnk-system.git
   cd pqnk-system
   ```
2. Create a Python virtual environment and install requirements:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   pip install gunicorn  # Critical for production WSGI
   ```
3. Create a `.env` file in the VM root folder with your API Keys (OpenAI, Qdrant, Google Drive credentials json, PostgreSQL connection).
4. Run the backend constantly using a `systemd` service file so it restarts if it crashes.

### Step E: Configure Nginx
Nginx will accept global internet requests on port 80 (HTTP) and route them to your internal Gunicorn app running on port 5000.
```bash
sudo nano /etc/nginx/sites-available/pqnk
```
```nginx
server {
    listen 80;
    server_name YOUR_VM_PUBLIC_IP;

    # Allow large file uploads for videos/PDFs
    client_max_body_size 100M; 

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```
Enable it and test Nginx.

### Step F: Deploy the React Frontend to Vercel
1. In your local VSCode, open `frontend-react/PQNK_Frontend/vite.config.js`. You will change the proxy from `localhost:5000` to point directly to your Azure VM IP address/Domain name. Alternatively, in production, use absolute URLs via `process.env.VITE_API_URL` inside your Axios calls.
2. Push your final code to GitHub.
3. Log into [Vercel.com](https://vercel.com/) and select "Add New Project".
4. Import your GitHub Repo.
5. Set the Framework Preset to **Vite**, and the Root Directory to `frontend-react/PQNK_Frontend`.
6. Add the Environment Variable: `VITE_API_URL` = `http://<YOUR_AZURE_VM_IP>`.
7. Click **Deploy**. Vercel will give you a public URL (e.g., `https://pqnk-frontend.vercel.app`).

### Step G: Final Linkage (CORS)
In your Azure VM, update `app.py` so the `CORS` configuration allows requests from your new Vercel domain, replacing `localhost:5173`. Restart the Gunicorn service.

---

## 4. Summary

**To successfully deploy, you will:**
1. Put Frontend on Vercel.
2. Put Backend + Postgres database on an Azure Ubuntu Linux VM.
3. Adjust frontend Axios calls to point to the VM's public IP instead of localhost.
4. Adjust Backend CORS to allow requests from the Vercel domain instead of localhost.
5. Use Nginx and Gunicorn to serve Flask instead of the debug runner.
6. Continue using the cloud Qdrant Vector database as you are doing now.
