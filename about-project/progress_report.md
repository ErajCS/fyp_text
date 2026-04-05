# PQNK Platform: Comprehensive Progress Update

## 1. Executive Summary
The PQNK Platform has evolved from a local prototype into a highly robust, multi-modal, and secure web application. The core objective of the platform is to serve as an intelligent, bilingual (Urdu/English) agricultural knowledge base. Recent development efforts have focused on stabilizing the architecture, hardening security, completing the automated multimodal data ingestion pipeline, and establishing enterprise-grade CI/CD practices.

---

## 2. Infrastructure & Repository Management
The application follows a modern decoupled architecture:
*   **Frontend:** React.js powered by Vite, utilizing Tailwind CSS for styling and Framer Motion for animations.
*   **Backend:** Flask (Python) exposing RESTful APIs.
*   **Databases:** PostgreSQL handles structured relational data (Users, Roles, File Metadata), while **Qdrant** acts as the high-dimensional Vector Database for semantic search.

### Google Drive Integration (`drive_service.py`)
To ensure high availability and cloud redundancy, the repository is directly synced with Google Drive.
*   **API Usage:** Utilizes the Google Drive API via OAuth2 Service Accounts (`google-api-python-client`, `google-auth`).
*   **Automated Sync:** When a user uploads a document through the PQNK dashboard, it is stored locally, but a background thread immediately triggers `MediaIoBaseUpload` to push a copy to a dedicated Google Drive folder.
*   **Deletion & Addition Logic:** The system maintains parity between the local repository and the cloud. If an administrator deletes a file from the PQNK UI, a request is made to the Drive API to trash the corresponding file ID, and the metadata is purged from PostgreSQL. Adding files automatically updates both databases (relational + vector) and the Drive cloud storage seamlessly.

---

## 3. The Automated Multimodal Pipeline (`pipeline_service.py`)
The standout feature of the backend architecture is the completely automated data ingestion pipeline. When a file is uploaded, the backend immediately returns a success status to the user while spawning background threads (`threading.Thread`) to process the raw file into AI-searchable embeddings.

The pipeline natively supports four distinct file types:
1.  **Documents (PDF/DOCX/TXT/CSV):**
    *   Text is extracted natively. If the document is an image-heavy PDF, PyTesseract OCR is applied.
    *   **Embedded Image Extraction:** A custom module (`extracting_images_from_pdfs.py`) scans PDFs and pulls out embedded charts, graphs, and photos.
2.  **Images:**
    *   Passed through `translating_images.py`, which leverages GPT-4o-mini Vision to comprehensively analyze the image and generate a highly detailed bilingual (English and Urdu) textual description of the visual data.
3.  **Video:**
    *   Audio tracks are stripped from the video file locally using `FFmpeg`.
    *   The extracted audio is transcribed using Whisper.
4.  **Audio (MP3/WAV):**
    *   Processed directly using `faster-whisper`. To ensure broad compatibility across production environments (which may lack NVIDIA CUDA GPUs), the model is forced to execute via `device="cpu"` and `compute_type="int8"`.
    *   Audio files are marked as "AI-internal" and are intentionally hidden from the visual UI repository view, while their detailed transcripts heavily power the RAG chatbot.

After the file is converted into text, `ingest_data.py` chunks the text, creates vectors using OpenAI (`text-embedding-3-small`), and pushes them to Qdrant. Finally, `ingest_to_postgres.py` logs the file's metadata into the relational database.

---

## 4. Chatbot RAG Intelligence
The PQNK Chatbot is an intelligent Retrieval-Augmented Generation (RAG) agent capable of conversational memory.
*   **Bilingual Voice Input:** Users can toggle between English and Urdu. The frontend utilizes the browser's native `SpeechRecognition` Web API to allow farmers to speak directly to the bot. 
*   **Synthesized Sourcing:** Standard RAG systems provide plain text. The PQNK backend Pipeline includes a `_enrich_source()` function that maps vector chunks back to their original database entries. 
*   **Internal Routing:** Instead of sending users away to external Google Drive URLs, the chatbot injects clickable internal links (e.g., `/api/repository/file/Video1_translated.mp4`). This allows users to read an AI answer, click the source link, and watch the exact corresponding video directly within the PQNK platform.
*   **Markdown Parsing Fixes:** Upgraded the React Markdown renderer to safely handle complex paragraph breaks, ensuring numbered lists (1, 2, 3...) do not reset back to `1` when rendering complex GPT-4 outputs.

---

## 5. Security Hardening
Significant work was put into securing the platform against unauthorized access and malicious inputs:
*   **Strict Access Control:** The frontend utilizes a `ProtectedRoute` React wrapper that verifies JWT/Sessions against an allowed array of roles (e.g., `["superadmin", "admin"]`). Bypassing the URL bar is impossible. On the backend, custom Flask decorators (`@require_role`) reject unauthorized REST requests.
*   **Password Policies:** A stringent `validate_password()` function enforces an 8+ character minimum containing an uppercase letter, lowercase letter, digit, and special symbol.
*   **Multi-Channel OTP Verification Management:** A fully custom One-Time Password (OTP) system was built from scratch to verify user identity during Registration and Forgot Password workflows.
    *   **Generation & Database Storage:** When a user signs up or requests a password reset, `secrets.randbelow(1000000)` generates a cryptographically secure 6-digit code. This code is stored in the PostgreSQL database under the user's `otp_code` column, alongside an `otp_expiry` timestamp set to exactly 10 minutes in the future (`datetime.utcnow() + timedelta(minutes=10)`).
    *   **Concurrent Dual-Channel Delivery:** To ensure the user gets the code as fast as possible without freezing the frontend, the backend initiates two parallel background threads (`threading.Thread`). 
        1.   **Thread 1 (Email):** Connects to the Gmail SMTP server using `smtplib` and `email.mime` to dispatch a beautifully formatted HTML email containing the OTP.
        2.   **Thread 2 (SMS):** Simultaneously connects to the Twilio REST API (`twilio.rest`) to dispatch the same OTP to the user's registered smartphone via SMS. 
    *   **Verification & Expiration Logic (`/api/verify-otp`):** When the user submits the code, the backend first checks if `datetime.utcnow() > user.otp_expiry`. If it has expired, the request is rejected with a 400 Bad Request. If it matches, `is_verified` is flipped to `True`, and the OTP columns are immediately set to `None` in the database to prevent replay attacks. We also implemented a `/api/resend-otp` endpoint using the exact same threaded delivery logic if the user misses the first prompt.
*   **Rate-Limiting:** To prevent attackers from spamming emails, an IP-based rate limiter restricts calls to endpoints like `/api/forgot-password`.

---

## 6. CI/CD Pipeline Implementation (GitHub Actions)
To transition from the development phase to production readiness, an automated Continuous Integration pipeline (`ci.yml`) was established on GitHub.
Every push to any branch kicks off two automated virtual machines (Ubuntu latest):

1.  **Frontend Pipeline (Node.js 20):** 
    *   Downloads dependencies completely cleanly (`npm install`).
    *   Runs customized ESLint checks. We configured ESLint to act as a safeguard (catching real logic errors) without acting as a blocker (ignoring unused Framer Motion animation variables or intentional empty `catch` blocks).
    *   Verifies that the application can successfully bundle for production via Vite.
    
2.  **Backend Pipeline (Python 3.11):**
    *   Creates a standalone virtual environment and installs all dependencies (`qdrant-client`, `faster-whisper`, `openai`, etc.).
    *   Executes 15 comprehensive Pytest suites spanning `test_auth.py`, `test_integration.py`, and `test_api.py`.
    *   **In-Memory DB Mocking:** The CI pipeline is written intelligently to intercept the Flask application load process, wiping out PostgreSQL dependency variables and injecting `sqlite:///:memory:`. 
    *   **External Integration Validations:** The pipeline securely accesses GitHub Environment Secrets to make a live handshake with both Qdrant and the OpenAI API, confirming network health before giving a green checkmark.

---

## 7. UI/UX Refinements
The user interface underwent significant iteration to provide a premium, modern experience appropriate for a professional agricultural platform:
*   **Thematic Overhaul:** Transitioned from a stark, all-white aesthetic to a professional, high-contrast dark emerald theme. The sidebar and navigation components were heavily refined for better accessibility.
*   **Authentication Flow Design:** Implemented a modern "split-panel" design for the Login and Signup pages, incorporating visually striking agricultural field backgrounds to ground the platform's identity.
*   **Structured Layouts:** Fixed overlapping card issues in the Profile page by utilizing CSS grid structures (`col-span-1` vs `col-span-2`), enabling dynamic user data collection forms to render cleanly.
*   **Comprehensive Data Collection:** Integrated detailed user metadata forms (e.g., Designation, Organization) into the signup and profile workflows to better categorize users (researchers vs. farmers).
*   **Polished Navigation & Structure:** Fully implemented and linked public-facing informative pages, including standardizing the *About Us* and *Contact Us* sections, as well as anchoring the application with a professional, cohesive footer.
