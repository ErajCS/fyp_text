# PQNK Platform: Comprehensive Progress Update

## 1. Executive Summary
The PQNK Platform has evolved from a local prototype into a highly robust, multi-modal, and secure web application. The core objective of the platform is to serve as an intelligent, bilingual (Urdu/English) agricultural knowledge base. Recent development efforts have focused on stabilizing the architecture, hardening security, completing the automated multimodal data ingestion pipeline, and establishing enterprise-grade CI/CD practices.

---

## 2. Infrastructure & Repository Management
The application features a decoupled architecture designed for high availability and synchronized storage:
*   **Frontend:** React.js powered by Vite.
*   **Backend:** Flask (Python) exposing RESTful APIs.
*   **Databases:** PostgreSQL (relational) and Qdrant (vector).

### Repository Setup & Frontend Viewing Logic
The repository serves as the central hub where all agricultural documents, audio, and videos are accessible.
*   **Setup:** We utilize a PostgreSQL database managed via SQLAlchemy. When a user navigates to the repository on the frontend (`BrowseRepository.jsx`), a `GET /api/resources` API call is made. 
*   **The Backend Controller (`app.py`):** The backend queries the `Resource` table, supporting pagination, category filtering, and text search. It returns metadata (title, category, upload date) to the frontend.
*   **Frontend Rendering:** The React frontend maps this data into interactive cards. When a user clicks a file, the frontend calls the `/api/repository/file/<filename>` endpoint. 
*   **File Serving Logic:** To view the content directly without leaving the platform, the Flask backend securely uses `send_from_directory` to serve the physical file (PDF, MP4, etc.) from the server's local storage folder.

### Google Drive Synchronization (`drive_service.py`)
To ensure enterprise-grade disaster recovery and cloud resilience, every file in the system is securely duplicated to Google Drive.
*   **Authentication & Initialization:** We established an OAuth2 Google Service Account (`google_service_account.json`). The `drive_service.py` script authenticates headlessly without any manual browser prompts.
*   **Seamless Uploads (`MediaIoBaseUpload`):** When a file is uploaded to the PQNK platform, it is saved locally first. Instantly, `drive_service.py` spawns a background process that reads the file stream and uses `MediaIoBaseUpload` to quietly push it to a specified shared Google Drive folder.
*   **Symmetric Deletion:** If an admin deletes a resource from the React dashboard, the backend triggers `.files().delete(fileId=drive_file_id).execute()` via the Drive API to trash it in the cloud, while simultaneously purging the local file and PostgreSQL record, ensuring precise 1:1 parity between local storage and cloud storage.

---

## 3. The Automated Multimodal Pipeline
A cornerstone of the PQNK platform is the completely automated data ingestion pipeline (`pipeline_service.py`). When an administrator uploads a new image, video, or PDF, the UI immediately displays a success message. Concurrently, the backend spawns a `threading.Thread` calling `pipeline_service.py` so the user is never stuck waiting on a loading screen while AI processing occurs in the background.

The pipeline comprises a series of highly specialized Python scripts, running in sequence depending on the file type:

1.  **Image Translation & Processing (`translating_images.py`)**
    *   **Logic:** When an image (e.g., an infographic showing crop disease) is uploaded, this script reads the file bytes, encodes it to Base64, and sends it to the **GPT-4o-mini Vision API**.
    *   **Result:** It returns a detailed, bilingual (Urdu and English) textual description of everything visible in the image, saving it to `/pipeline_workspace` as a `.txt` file ready for vectorization.

2.  **PDF/Document Analysis & Extraction (`extracting_images_from_pdfs.py` & `extraction.py`)**
    *   **Logic:** Raw PDFs often trap valuable visual data. `extracting_images_from_pdfs.py` scans every page of a document using PyMuPDF. If it detects embedded charts or photographs, it extracts them, saves them locally, and automatically feeds them into `translating_images.py`.
    *   **Result:** The final `.txt` output contains both the raw text of the document AND full descriptions of its embedded images.

3.  **Video Translation & Stripping (`translating_video.py`)**
    *   **Logic:** Processing massive video files with AI directly is expensive and slow. This script uses the powerful `ffmpeg` utility locally to strip out just the audio track from the uploaded `.mp4`.
    *   **Result:** It extracts a lightweight MP3/WAV file, drastically speeding up the pipeline.

4.  **Audio Transcription (`local_whisper.py`)**
    *   **Logic:** Audio tracks (from native uploads or stripped videos) are fed into `faster-whisper`. Since cloud hosting environments often lack expensive NVIDIA GPUs, we hardcoded the model to run optimally on standard CPU hardware using `device="cpu"` and `compute_type="int8"`.
    *   **Result:** A pristine, timestamped transcript is generated. The audio file is then natively integrated to power the Chatbot.

5.  **Vectorization & Knowledge Injection (`ingest_data.py` & `ingest_to_postgres.py`)**
    *   Once the above scripts reduce any media file into a final `.txt` document, `ingest_data.py` takes over. 
    *   It recursively reads the text, splits it into semantic chunks using LangChain's `RecursiveCharacterTextSplitter`, generates dense numerical vectors via OpenAI's `text-embedding-3-small` endpoint, and pushes them into the **Qdrant Vector Database**. 
    *   Finally, the pipeline terminates successfully by calling `ingest_to_postgres.py`, which writes all final tracking metadata (upload dates, final paths, drive IDs) to the PostgreSQL database.

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
