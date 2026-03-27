# from flask import Flask, render_template, request, jsonify
# from rag_demo import rag_pipeline # Import your existing function

# app = Flask(__name__)

# # 1. Serve the HTML Interface
# @app.route("/")
# def home():
#     return render_template("chatbot.html")

# # 2. Handle the Chat Logic
# @app.route("/get_response", methods=["POST"])
# def get_response():
#     user_message = request.json.get("msg")
    
#     # Call your existing RAG pipeline
#     # This runs your retrieval + generation logic
#     ai_response = rag_pipeline(user_message)
    
#     return jsonify({"response": ai_response})

# if __name__ == "__main__":
#     # Open automatically in the browser
#     print("Starting AgriChat Interface...")
#     app.run(debug=True)















# from flask import Flask, render_template, request, jsonify, redirect, url_for
# from rag_demo import rag_pipeline

# app = Flask(__name__)

# # =======================
# # PAGE ROUTES
# # =======================

# @app.route("/")
# def login():
#     return render_template("login.html")

# @app.route("/signup")
# def signup():
#     return render_template("signup.html")

# @app.route("/dashboard")
# def dashboard():
#     return render_template("dashboard.html")

# @app.route("/chatbot")
# def chatbot():
#     return render_template("chatbot.html")

# # =======================
# # CHAT API
# # =======================

# @app.route("/get_response", methods=["POST"])
# def get_response():
#     user_message = request.json.get("msg")
#     ai_response = rag_pipeline(user_message)
#     return jsonify({"response": ai_response})

# if __name__ == "__main__":
#     print("Starting AgriChat Interface...")
#     app.run(debug=True)



















# import psycopg2
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
# from rag_demo import rag_pipeline

# app = Flask(__name__)
# app.config["SECRET_KEY"] = "supersecretkey"
# app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///users.db"

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = "login"

# # =======================
# # DATABASE MODEL
# # =======================

# class User(db.Model, UserMixin):
#     id = db.Column(db.Integer, primary_key=True)
#     name = db.Column(db.String(100))
#     email = db.Column(db.String(120), unique=True)
#     password = db.Column(db.String(200))
#     dark_mode = db.Column(db.Boolean, default=False)

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# # =======================
# # ROUTES
# # =======================

# @app.route("/", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form["email"]
#         password = request.form["password"]

#         user = User.query.filter_by(email=email).first()

#         if user and bcrypt.check_password_hash(user.password, password):
#             login_user(user)
#             return redirect("/dashboard")

#         return render_template("login.html", error="Invalid email or password")

#     return render_template("login.html")

# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     if request.method == "POST":
#         name = request.form["name"]
#         email = request.form["email"]
#         password = bcrypt.generate_password_hash(request.form["password"]).decode("utf-8")

#         if User.query.filter_by(email=email).first():
#             return render_template("signup.html", error="Email already exists")

#         new_user = User(name=name, email=email, password=password)
#         db.session.add(new_user)
#         db.session.commit()

#         return redirect("/")

#     return render_template("signup.html")

# @app.route("/dashboard")
# @login_required
# def dashboard():
#     return render_template("dashboard.html", user=current_user)

# @app.route("/toggle_theme")
# @login_required
# def toggle_theme():
#     current_user.dark_mode = not current_user.dark_mode
#     db.session.commit()
#     return redirect(request.referrer)

# @app.route("/logout")
# def logout():
#     logout_user()
#     return redirect("/")

# # =======================
# # CHATBOT ROUTES
# # =======================

# @app.route("/chatbot")
# @login_required
# def chatbot():
#     return render_template("chatbot.html", user=current_user)

# @app.route("/get_response", methods=["POST"])
# @login_required
# def get_response():
#     user_message = request.json.get("msg")
#     ai_response = rag_pipeline(user_message)
#     return jsonify({"response": ai_response})

# # =======================
# # RUN
# # =======================

# if __name__ == "__main__":
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)





































# import psycopg2
# from werkzeug.security import generate_password_hash, check_password_hash
# from flask import Flask, render_template, request, jsonify, redirect, url_for, session
# from flask_sqlalchemy import SQLAlchemy
# from flask_bcrypt import Bcrypt
# from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
# from rag_demo import rag_pipeline
# from datetime import datetime

# # =======================
# # FLASK APP CONFIG
# # =======================
# app = Flask(__name__)
# app.config["SECRET_KEY"] = "supersecretkey"

# # PostgreSQL connection URI (update username/password/dbname as needed)
# app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:admin123@localhost:5432/pqnk_db"
# app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# db = SQLAlchemy(app)
# bcrypt = Bcrypt(app)

# login_manager = LoginManager()
# login_manager.init_app(app)
# login_manager.login_view = "login"

# # =======================
# # DATABASE MODEL
# # =======================
# class User(db.Model, UserMixin):
#     __tablename__ = 'users'
#     id = db.Column('user_id', db.Integer, primary_key=True)
#     name = db.Column('full_name', db.String(100), nullable=False)
#     email = db.Column(db.String(255), unique=True, nullable=False)
#     phone = db.Column(db.String(20))  # new field
#     password = db.Column('password_hash', db.Text, nullable=False)
#     role = db.Column(db.String(20), nullable=False, default='seeker')
#     created_at = db.Column(db.DateTime, default=datetime.utcnow)
#     is_verified = db.Column(db.Boolean, default=False)
#     otp_code = db.Column(db.String(6))
#     otp_expiry = db.Column(db.DateTime)
#     dark_mode = db.Column(db.Boolean, default=False)

# @login_manager.user_loader
# def load_user(user_id):
#     return User.query.get(int(user_id))

# # =======================
# # ROUTES
# # =======================

# @app.route("/", methods=["GET", "POST"])
# def login():
#     if request.method == "POST":
#         email = request.form["email"]
#         password = request.form["password"]

#         user = User.query.filter_by(email=email).first()

#         if user and bcrypt.check_password_hash(user.password, password):
#             login_user(user)
#             return redirect("/dashboard")

#         return render_template("login.html", error="Invalid email or password")

#     return render_template("login.html")


# @app.route("/signup", methods=["GET", "POST"])
# def signup():
#     error = None
#     success = None

#     if request.method == "POST":
#         name = request.form["name"]
#         email = request.form["email"]
#         phone = request.form["phone"]
#         password_raw = request.form["password"]
#         confirm_password = request.form["confirm_password"]

#         # Password match validation
#         if password_raw != confirm_password:
#             error = "Passwords do not match."
#             return render_template("signup.html", error=error)

#         # Check if email already exists
#         if User.query.filter_by(email=email).first():
#             error = "Email already exists."
#             return render_template("signup.html", error=error)

#         # Hash password
#         password_hashed = bcrypt.generate_password_hash(password_raw).decode("utf-8")

#         # Save new user
#         new_user = User(
#             name=name,
#             email=email,
#             phone=phone,
#             password=password_hashed,
#             role='seeker',        # default role
#             dark_mode=False
#         )

#         db.session.add(new_user)
#         db.session.commit()

#         success = "Account created successfully! You may now login."
#         return render_template("signup.html", success=success)

#     return render_template("signup.html")


# @app.route("/dashboard")
# @login_required
# def dashboard():
#     return render_template("dashboard.html", user=current_user)


# @app.route("/toggle_theme")
# @login_required
# def toggle_theme():
#     current_user.dark_mode = not current_user.dark_mode
#     db.session.commit()
#     return redirect(request.referrer)


# @app.route("/logout")
# @login_required
# def logout():
#     logout_user()
#     return redirect("/")


# # =======================
# # CHATBOT ROUTES
# # =======================
# @app.route("/chatbot")
# @login_required
# def chatbot():
#     return render_template("chatbot.html", user=current_user)


# @app.route("/get_response", methods=["POST"])
# @login_required
# def get_response():
#     user_message = request.json.get("msg")
#     ai_response = rag_pipeline(user_message)
#     return jsonify({"response": ai_response})


# # =======================
# # RUN
# # =======================
# if __name__ == "__main__":
#     with app.app_context():
#         db.create_all()
#     app.run(debug=True)















































import psycopg2
import asyncio
import edge_tts
import uuid
import os
import re

# ── Load .env FIRST — before importing rag_demo which reads OPENAI_API_KEY ──
try:
    from dotenv import load_dotenv
    _env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
    load_dotenv(_env_path, override=True)
    print("✅ .env loaded")
except ImportError:
    pass

from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from rag_demo import rag_pipeline, detect_lang
from datetime import datetime


# =======================
# FLASK APP CONFIG
# =======================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, '..', 'static')


app = Flask(__name__, static_folder=STATIC_DIR)
# Import and enable CORS
from flask_cors import CORS
CORS(app, supports_credentials=True, origins=["http://localhost:5173", "http://127.0.0.1:5173"])

app.config["SECRET_KEY"] = "supersecretkey"
# PostgreSQL connection URI
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:admin123@localhost:5432/pqnk_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Create folder for audio files
AUDIO_DIR = os.path.join(STATIC_DIR, "audio")
os.makedirs(AUDIO_DIR, exist_ok=True)

db = SQLAlchemy(app)
bcrypt = Bcrypt(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = "login"

# =======================
# DATABASE MODEL
# =======================
class User(db.Model, UserMixin):
    __tablename__ = 'users'
    id = db.Column('user_id', db.Integer, primary_key=True)
    name = db.Column('full_name', db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    password = db.Column('password_hash', db.Text, nullable=False)
    role = db.Column(db.String(20), nullable=False, default='seeker')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_verified = db.Column(db.Boolean, default=False)
    otp_code = db.Column(db.String(6))
    otp_expiry = db.Column(db.DateTime)
    dark_mode = db.Column(db.Boolean, default=False)


class Resource(db.Model):
    """Repository item: document, image, or video."""
    __tablename__ = 'resources'
    id          = db.Column(db.Integer, primary_key=True)
    title       = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text)
    category    = db.Column(db.String(100), nullable=False, default='General')
    keywords    = db.Column(db.String(500))          # comma-separated
    file_type   = db.Column(db.String(20), nullable=False)  # document / image / video
    filename    = db.Column(db.String(255))          # stored filename on disk (null for video-link-only)
    original_name = db.Column(db.String(255))        # original upload name
    video_link  = db.Column(db.String(512))          # YouTube / external link (videos only)
    uploaded_by = db.Column(db.Integer, db.ForeignKey('users.user_id'))
    created_at  = db.Column(db.DateTime, default=datetime.utcnow)
    uploader    = db.relationship('User', foreign_keys=[uploaded_by])


# Directory for uploaded files
UPLOAD_DIR = os.path.join(BASE_DIR, '..', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)
ALLOWED_DOCS   = {'pdf', 'doc', 'docx', 'txt', 'pptx', 'xlsx'}
ALLOWED_IMAGES = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'svg'}
ALLOWED_VIDEOS = {'mp4', 'webm', 'mov', 'avi', 'mkv'}

def allowed_file(filename, file_type):
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    if file_type == 'document': return ext in ALLOWED_DOCS
    if file_type == 'image':    return ext in ALLOWED_IMAGES
    if file_type == 'video':    return ext in ALLOWED_VIDEOS
    return False


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))



# =======================
async def generate_speech_file(text, voice, output_path):
    communicate = edge_tts.Communicate(text, voice)
    await communicate.save(output_path)

def clean_text_for_audio(text):
    # Remove Markdown (**, ###)
    text = text.replace("*", "").replace("#", "")
    # Remove Sources section
    if "### Sources" in text:
        text = text.split("### Sources")[0]
    if "### حوالہ جات" in text:
        text = text.split("### حوالہ جات")[0]
    return text

# =======================
# ROUTES
# =======================

@app.route("/", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form["email"]
        password = request.form["password"]
        user = User.query.filter_by(email=email).first()
        if user and bcrypt.check_password_hash(user.password, password):
            login_user(user)
            return redirect("/dashboard")
        return render_template("login.html", error="Invalid email or password")
    return render_template("login.html")

@app.route("/signup", methods=["GET", "POST"])
def signup():
    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password_raw = request.form["password"]
        confirm_password = request.form["confirm_password"]
        
        if password_raw != confirm_password:
            return render_template("signup.html", error="Passwords do not match.")
        if User.query.filter_by(email=email).first():
            return render_template("signup.html", error="Email already exists.")
            
        password_hashed = bcrypt.generate_password_hash(password_raw).decode("utf-8")
        new_user = User(name=name, email=email, phone=phone, password=password_hashed)
        db.session.add(new_user)
        db.session.commit()
        return render_template("signup.html", success="Account created successfully!")
    return render_template("signup.html")

@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html", user=current_user)

@app.route("/toggle_theme")
@login_required
def toggle_theme():
    current_user.dark_mode = not current_user.dark_mode
    db.session.commit()
    return redirect(request.referrer)

@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect("/")

# =======================
# CHATBOT & AUDIO ROUTES
# =======================
@app.route("/chatbot")
@login_required
def chatbot():
    return render_template("chatbot.html", user=current_user)

@app.route("/get_response", methods=["POST"])
@login_required
def get_response():
    user_message = request.json.get("msg", "").strip()
    if not user_message:
        return jsonify({"response": "Please enter a message."}), 400
    try:
        ai_response = rag_pipeline(user_message)
        return jsonify({"response": ai_response})
    except Exception as e:
        print(f"❌ RAG pipeline error: {e}")
        return jsonify({"response": "⚠️ Sorry, I encountered an error processing your request. Please try again."}), 500

@app.route("/generate_audio", methods=["POST"])
@login_required
def generate_audio():
    text = request.json.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    print(f"🎤 Request to read: {text[:50]}...") # Debug print

    # 1. Clean Text
    clean_text = clean_text_for_audio(text)

    # 2. Detect Language
    lang = detect_lang(clean_text)
    if lang == "ur":
        voice = "ur-PK-UzmaNeural"
    else:
        voice = "en-US-AriaNeural"

    # 3. Generate Filename
    filename = f"speech_{uuid.uuid4()}.mp3"
    filepath = os.path.join(AUDIO_DIR, filename)

    # 4. Generate Audio (FIXED ASYNC LOOP)
    # Use a new event loop to avoid conflict with Flask's thread
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(generate_speech_file(clean_text, voice, filepath))
        loop.close()
    except Exception as e:
        print(f"❌ TTS Error: {e}")
        return jsonify({"error": str(e)}), 500

    # 5. Return URL
    audio_url = url_for('static', filename=f'audio/{filename}')
    print(f"✅ Audio generated: {audio_url}")
    
    return jsonify({"audio_url": audio_url})

# =======================
# JSON API FOR REACT FRONTEND
# =======================

import secrets
import smtplib
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Load .env each time the module is used (ensures fresh values on restart)
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env'), override=True)
    print("✅ .env loaded")
except ImportError:
    pass

# ─── Phone number normalizer ──────────────────────────────────────────────────
def normalize_phone(phone: str) -> str:
    """
    Converts Pakistani phone numbers to E.164 format for Twilio.
    Examples:  03001234567  →  +923001234567
               923001234567 →  +923001234567
               +923001234567 → +923001234567
    """
    phone = phone.strip().replace(" ", "").replace("-", "")
    if phone.startswith("0"):
        phone = "+92" + phone[1:]
    elif phone.startswith("92") and not phone.startswith("+"):
        phone = "+" + phone
    elif not phone.startswith("+"):
        phone = "+" + phone
    return phone

# ─── SMS via Twilio ───────────────────────────────────────────────────────────
def send_otp_sms(to_phone: str, otp_code: str, name: str = "") -> tuple:
    """Send OTP via SMS using Twilio. Returns (True, None) or (False, error_msg)."""
    sid   = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    from_ = os.getenv("TWILIO_FROM_NUMBER", "").strip()

    if not to_phone or not sid or not token or not from_ \
            or sid == "your_account_sid_here" or token == "your_auth_token_here":
        print(f"📵 SMS not configured (Twilio creds missing) — skipping SMS for {to_phone}")
        return False, "SMS not configured"

    formatted = normalize_phone(to_phone)
    print(f"📱 Sending SMS OTP to {formatted}…")

    try:
        from twilio.rest import Client
        from twilio.base.exceptions import TwilioRestException
        client = Client(sid, token)
        body   = (
            f"AgriChat – PQNK Knowledge System\n"
            f"Hi {name or 'there'}! Your verification code is: {otp_code}\n"
            f"This code expires in 10 minutes. Do NOT share it."
        )
        message = client.messages.create(body=body, from_=from_, to=formatted)
        print(f"✅ SMS sent to {formatted} (SID: {message.sid})")
        return True, None
    except TwilioRestException as e:
        err = f"Twilio error {e.code}: {e.msg}"
        print(f"❌ {err}")
        return False, err
    except ImportError:
        err = "Twilio package not installed. Run: pip install twilio"
        print(f"❌ {err}")
        return False, err
    except Exception as e:
        print(f"❌ SMS send error: {e}")
        return False, str(e)

# ─── Email via Gmail SMTP ─────────────────────────────────────────────────────
def send_otp_email(to_email: str, otp_code: str, name: str = "") -> tuple:
    """Send OTP email via Gmail SMTP. Returns (True, None) or (False, error_msg)."""
    mail_user = os.getenv("MAIL_USERNAME", "").strip()
    mail_pass = os.getenv("MAIL_PASSWORD", "").strip()
    mail_name = os.getenv("MAIL_FROM_NAME", "AgriChat PQNK").strip()

    print(f"\n📧 Email config → user='{mail_user}' pass_length={len(mail_pass)}")

    if not mail_user or not mail_pass or mail_user == "your_gmail@gmail.com" or len(mail_pass) < 10:
        print(f"\n{'='*50}\n🔑 OTP FOR {to_email}: {otp_code}\n{'='*50}\n")
        return True, None

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"AgriChat – Your Verification Code: {otp_code}"
        msg["From"]    = f"{mail_name} <{mail_user}>"
        msg["To"]      = to_email

        html_body = f"""
        <div style="font-family:Arial,sans-serif;max-width:480px;margin:auto;background:#f9fafb;border-radius:12px;overflow:hidden;border:1px solid #e5e7eb">
          <div style="background:linear-gradient(135deg,#064e3b,#065f46);padding:28px 32px;text-align:center">
            <h1 style="color:#fff;margin:0;font-size:22px">AgriChat</h1>
            <p style="color:#6ee7b7;margin:4px 0 0;font-size:12px;letter-spacing:2px;text-transform:uppercase">PQNK Knowledge System</p>
          </div>
          <div style="padding:32px">
            <p style="color:#374151;font-size:15px">Hi <strong>{name or 'there'}</strong>,</p>
            <p style="color:#6b7280;font-size:14px">Your verification code for AgriChat is:</p>
            <div style="text-align:center;margin:24px 0">
              <span style="display:inline-block;background:#064e3b;color:#fff;font-size:36px;font-weight:bold;letter-spacing:10px;padding:16px 32px;border-radius:12px">{otp_code}</span>
            </div>
            <p style="color:#9ca3af;font-size:13px;text-align:center">This code expires in <strong>10 minutes</strong>. Do not share it with anyone.</p>
          </div>
          <div style="background:#f3f4f6;padding:16px 32px;text-align:center">
            <p style="color:#9ca3af;font-size:12px;margin:0">2026 AgriChat - PQNK Agriculture Repository</p>
          </div>
        </div>
        """
        msg.attach(MIMEText(html_body, "html"))

        print("📤 Connecting to Gmail SMTP…")
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(mail_user, mail_pass)
            server.sendmail(mail_user, to_email, msg.as_string())
        print(f"✅ OTP email sent to {to_email}")
        return True, None
    except smtplib.SMTPAuthenticationError as e:
        msg_txt = "Gmail auth failed — check MAIL_PASSWORD is a valid 16-char App Password"
        print(f"\n❌ SMTP Auth Error: {e}\n⚠️  {msg_txt}\n🔑 Fallback OTP: {otp_code}\n")
        return False, msg_txt
    except Exception as e:
        print(f"\n❌ Email send error: {e}\n🔑 Fallback OTP: {otp_code}\n")
        return False, str(e)

# ─── Combined dispatcher (parallel threads) ───────────────────────────────────
def send_otp_both(email: str, phone: str, otp_code: str, name: str = "") -> dict:
    """
    Sends OTP via email AND SMS simultaneously.
    Returns: { "email": True/False, "sms": True/False, "sms_skipped": bool }
    """
    results = {}

    def _email():
        ok, _ = send_otp_email(email, otp_code, name)
        results["email"] = ok

        
    def _sms():
        if phone:
            ok, _ = send_otp_sms(phone, otp_code, name)
            results["sms"] = ok
            results["sms_skipped"] = False
        else:
            results["sms"] = False
            results["sms_skipped"] = True

    t1 = threading.Thread(target=_email)
    t2 = threading.Thread(target=_sms)
    t1.start(); t2.start()
    t1.join();  t2.join()
    return results




@app.route("/api/login", methods=["POST"])
def api_login():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    email = data.get("email", "")
    password = data.get("password", "")

    user = User.query.filter_by(email=email).first()
    if user and bcrypt.check_password_hash(user.password, password):
        if not user.is_verified:
            return jsonify({"success": False, "message": "Please verify your email first", "needs_otp": True, "email": email}), 403
        login_user(user)
        return jsonify({
            "success": True,
            "user": {
                "id": user.id,
                "name": user.name,
                "email": user.email,
                "phone": user.phone or "",
                "role": user.role,
                "dark_mode": user.dark_mode,
            }
        })
    return jsonify({"success": False, "message": "Invalid email or password"}), 401


@app.route("/api/signup", methods=["POST"])
def api_signup():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    name = data.get("name", "").strip()
    email = data.get("email", "").strip().lower()
    phone = data.get("phone", "").strip()
    password_raw = data.get("password", "")
    confirm_password = data.get("confirm_password", password_raw)

    # Validate required fields
    if not name or not email or not password_raw:
        return jsonify({"success": False, "message": "Name, email, and password are required"}), 400

    if len(password_raw) < 8:
        return jsonify({"success": False, "message": "Password must be at least 8 characters"}), 400

    if password_raw != confirm_password:
        return jsonify({"success": False, "message": "Passwords do not match"}), 400

    # Check email uniqueness
    existing = User.query.filter_by(email=email).first()
    if existing:
        if not existing.is_verified:
            # Re-send OTP for unverified account
            otp = f"{secrets.randbelow(1000000):06d}"
            from datetime import timedelta
            existing.otp_code = otp
            existing.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
            try:
                db.session.commit()
                send_otp_both(email, existing.phone or "", otp, existing.name)
                return jsonify({"success": True, "message": "OTP resent. Please verify your email.", "email": email, "has_phone": bool(existing.phone)}), 200
            except Exception as e:
                db.session.rollback()
                return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500
        return jsonify({"success": False, "message": "An account with this email already exists"}), 409

    # Generate OTP
    otp = f"{secrets.randbelow(1000000):06d}"
    from datetime import timedelta
    otp_expiry = datetime.utcnow() + timedelta(minutes=10)

    # Hash password
    password_hashed = bcrypt.generate_password_hash(password_raw).decode("utf-8")

    # Create user (is_verified=False until OTP confirmed)
    new_user = User(
        name=name,
        email=email,
        phone=phone,
        password=password_hashed,
        role="seeker",
        is_verified=False,
        otp_code=otp,
        otp_expiry=otp_expiry,
    )

    try:
        db.session.add(new_user)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Signup DB error: {e}")
        return jsonify({"success": False, "message": f"Could not create account. Please try again. ({str(e)[:120]})"}), 500

    # Send OTP via email AND SMS simultaneously
    delivery = send_otp_both(email, phone, otp, name)
    channels = []
    if delivery.get("email"): channels.append("email")
    if delivery.get("sms"):   channels.append("SMS")

    if channels:
        ch_str = " and ".join(channels)
        return jsonify({"success": True, "message": f"Account created! OTP sent via {ch_str}.", "email": email, "has_phone": bool(phone)}), 201
    else:
        return jsonify({"success": True, "message": "Account created! Check the server console for your OTP code.", "email": email, "has_phone": bool(phone)}), 201


@app.route("/api/verify-otp", methods=["POST"])
def api_verify_otp():
    data = request.get_json()
    email = (data or {}).get("email", "").strip().lower()
    otp = (data or {}).get("otp", "").strip()

    if not email or not otp:
        return jsonify({"success": False, "message": "Email and OTP are required"}), 400

    user = User.query.filter_by(email=email).first()

    if not user:
        return jsonify({"success": False, "message": "Account not found"}), 404

    if user.is_verified:
        return jsonify({"success": True, "message": "Already verified. Please login."}), 200

    if not user.otp_code or not user.otp_expiry:
        return jsonify({"success": False, "message": "No OTP found. Please request a new one."}), 400

    if datetime.utcnow() > user.otp_expiry:
        return jsonify({"success": False, "message": "OTP has expired. Please request a new one."}), 400

    if user.otp_code != otp:
        return jsonify({"success": False, "message": "Incorrect OTP. Please try again."}), 400

    # Mark verified, clear OTP
    user.is_verified = True
    user.otp_code = None
    user.otp_expiry = None
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500

    return jsonify({"success": True, "message": "Email verified successfully! You can now log in."}), 200


@app.route("/api/resend-otp", methods=["POST"])
def api_resend_otp():
    data = request.get_json()
    email = (data or {}).get("email", "").strip().lower()

    if not email:
        return jsonify({"success": False, "message": "Email is required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user:
        return jsonify({"success": False, "message": "Account not found"}), 404

    if user.is_verified:
        return jsonify({"success": False, "message": "Account is already verified"}), 400

    otp = f"{secrets.randbelow(1000000):06d}"
    from datetime import timedelta
    user.otp_code = otp
    user.otp_expiry = datetime.utcnow() + timedelta(minutes=10)
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500

    send_otp_both(email, user.phone or "", otp, user.name)
    return jsonify({"success": True, "message": "A new OTP has been sent to your email and phone.", "has_phone": bool(user.phone)}), 200



@app.route("/api/user", methods=["GET"])
@login_required
def api_user():
    return jsonify({
        "id": current_user.id,
        "name": current_user.name,
        "email": current_user.email,
        "phone": current_user.phone or "",
        "role": current_user.role,
        "dark_mode": current_user.dark_mode,
        "created_at": current_user.created_at.strftime("%B %Y") if current_user.created_at else "",
    })


@app.route("/api/user/update", methods=["POST"])
@login_required
def api_user_update():
    data = request.get_json()
    if not data:
        return jsonify({"success": False, "message": "No data provided"}), 400

    name = data.get("name", "").strip()
    phone = data.get("phone", "").strip()
    current_password = data.get("current_password", "")
    new_password = data.get("new_password", "")

    if name:
        current_user.name = name
    if phone is not None:
        current_user.phone = phone

    if new_password:
        if not current_password:
            return jsonify({"success": False, "message": "Current password is required to set a new password"}), 400
        if not bcrypt.check_password_hash(current_user.password, current_password):
            return jsonify({"success": False, "message": "Current password is incorrect"}), 401
        current_user.password = bcrypt.generate_password_hash(new_password).decode("utf-8")

    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({"success": False, "message": f"Database error: {str(e)}"}), 500

    return jsonify({
        "success": True,
        "message": "Profile updated successfully",
        "user": {
            "id": current_user.id,
            "name": current_user.name,
            "email": current_user.email,
            "phone": current_user.phone or "",
            "role": current_user.role,
            "dark_mode": current_user.dark_mode,
            "created_at": current_user.created_at.strftime("%B %Y") if current_user.created_at else "",
        }
    })


@app.route("/api/logout", methods=["POST"])
@login_required
def api_logout():
    logout_user()
    return jsonify({"success": True, "message": "Logged out"})


# =======================
# ADMIN JSON API
# =======================

def require_role(*roles):
    """Decorator: only allow users whose role is in `roles`."""
    from functools import wraps
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            if not current_user.is_authenticated:
                return jsonify({"success": False, "message": "Authentication required"}), 401
            if current_user.role not in roles:
                return jsonify({"success": False, "message": "Forbidden: insufficient permissions"}), 403
            return fn(*args, **kwargs)
        return wrapper
    return decorator


@app.route("/api/admin/stats", methods=["GET"])
@login_required
@require_role("admin", "superadmin")
def api_admin_stats():
    """Real-time platform statistics from PostgreSQL."""
    total_users   = User.query.count()
    verified      = User.query.filter_by(is_verified=True).count()
    unverified    = User.query.filter_by(is_verified=False).count()
    admins        = User.query.filter(User.role.in_(["admin", "superadmin"])).count()
    farmers       = User.query.filter_by(role="farmer").count()
    seekers       = User.query.filter_by(role="seeker").count()
    researchers   = User.query.filter_by(role="researcher").count()
    # Recent registrations in last 7 days
    from datetime import timedelta
    week_ago      = datetime.utcnow() - timedelta(days=7)
    new_this_week = User.query.filter(User.created_at >= week_ago).count()

    return jsonify({
        "success": True,
        "stats": {
            "total_users":    total_users,
            "verified":       verified,
            "unverified":     unverified,
            "admins":         admins,
            "farmers":        farmers,
            "seekers":        seekers,
            "researchers":    researchers,
            "new_this_week":  new_this_week,
        }
    })


@app.route("/api/admin/users", methods=["GET"])
@login_required
@require_role("admin", "superadmin")
def api_admin_users():
    """
    Paginated user list with optional filters.
    Query params: page (default 1), per_page (default 20), role, search
    """
    page     = int(request.args.get("page", 1))
    per_page = min(int(request.args.get("per_page", 20)), 100)
    role     = request.args.get("role", "")
    search   = request.args.get("search", "").strip()

    query = User.query
    if role:
        query = query.filter_by(role=role)
    if search:
        query = query.filter(
            (User.name.ilike(f"%{search}%")) | (User.email.ilike(f"%{search}%"))
        )

    total   = query.count()
    users   = query.order_by(User.created_at.desc()) \
                   .offset((page - 1) * per_page).limit(per_page).all()

    return jsonify({
        "success": True,
        "total":   total,
        "page":    page,
        "per_page": per_page,
        "users": [{
            "id":          u.id,
            "name":        u.name,
            "email":       u.email,
            "phone":       u.phone or "",
            "role":        u.role,
            "is_verified": u.is_verified,
            "created_at":  u.created_at.strftime("%b %d, %Y") if u.created_at else "—",
        } for u in users]
    })


@app.route("/api/admin/users/<int:user_id>/role", methods=["PATCH"])
@login_required
@require_role("admin", "superadmin")
def api_admin_change_role(user_id):
    """Change a user's role. Super admins can promote to admin; admins cannot."""
    data     = request.get_json() or {}
    new_role = data.get("role", "").strip().lower()
    VALID    = {"seeker", "farmer", "researcher", "admin", "superadmin"}
    if new_role not in VALID:
        return jsonify({"success": False, "message": f"Invalid role: {new_role}"}), 400

    # Only superadmin can grant admin/superadmin role
    if new_role in {"admin", "superadmin"} and current_user.role != "superadmin":
        return jsonify({"success": False, "message": "Only Super Admin can assign admin roles"}), 403

    target = User.query.get(user_id)
    if not target:
        return jsonify({"success": False, "message": "User not found"}), 404
    if target.id == current_user.id:
        return jsonify({"success": False, "message": "Cannot change your own role"}), 400

    target.role = new_role
    db.session.commit()
    return jsonify({"success": True, "message": f"Role updated to {new_role}", "user_id": user_id})


@app.route("/api/admin/users/<int:user_id>", methods=["DELETE"])
@login_required
@require_role("superadmin")
def api_admin_delete_user(user_id):
    """Super admin only: hard delete a user account."""
    target = User.query.get(user_id)
    if not target:
        return jsonify({"success": False, "message": "User not found"}), 404
    if target.id == current_user.id:
        return jsonify({"success": False, "message": "Cannot delete your own account"}), 400
    db.session.delete(target)
    db.session.commit()
    return jsonify({"success": True, "message": "User deleted"})


# =======================
# REPOSITORY API
# =======================

def resource_to_dict(r):
    return {
        "id":            r.id,
        "title":         r.title,
        "description":   r.description or "",
        "category":      r.category,
        "keywords":      r.keywords or "",
        "file_type":     r.file_type,
        "filename":      r.filename or "",
        "original_name": r.original_name or "",
        "video_link":    r.video_link or "",
        "uploaded_by":   r.uploader.name if r.uploader else "Unknown",
        "created_at":    r.created_at.strftime("%b %d, %Y") if r.created_at else "",
    }


@app.route("/api/repository", methods=["GET"])
@login_required
def api_repo_list():
    """List repository resources with optional filters."""
    q         = request.args.get("q", "").strip()
    category  = request.args.get("category", "")
    file_type = request.args.get("file_type", "")
    page      = int(request.args.get("page", 1))
    per_page  = min(int(request.args.get("per_page", 20)), 100)

    query = Resource.query
    if q:
        query = query.filter(
            (Resource.title.ilike(f"%{q}%")) |
            (Resource.keywords.ilike(f"%{q}%")) |
            (Resource.description.ilike(f"%{q}%"))
        )
    if category:
        query = query.filter_by(category=category)
    if file_type:
        query = query.filter_by(file_type=file_type)

    total     = query.count()
    resources = query.order_by(Resource.created_at.desc()) \
                     .offset((page - 1) * per_page).limit(per_page).all()

    # Get unique categories for filter dropdown
    cats = db.session.query(Resource.category).distinct().all()
    categories = sorted([c[0] for c in cats if c[0]])

    return jsonify({
        "success":    True,
        "total":      total,
        "page":       page,
        "per_page":   per_page,
        "categories": categories,
        "resources":  [resource_to_dict(r) for r in resources],
    })


@app.route("/api/repository/upload", methods=["POST"])
@login_required
@require_role("admin", "superadmin")
def api_repo_upload():
    """Upload a new resource (multipart form)."""
    import werkzeug.utils as wu

    title       = request.form.get("title", "").strip()
    description = request.form.get("description", "").strip()
    category    = request.form.get("category", "General").strip()
    keywords    = request.form.get("keywords", "").strip()
    file_type   = request.form.get("file_type", "").strip().lower()  # document / image / video
    video_link  = request.form.get("video_link", "").strip()

    if not title:
        return jsonify({"success": False, "message": "Title is required"}), 400
    if file_type not in ("document", "image", "video"):
        return jsonify({"success": False, "message": "file_type must be document, image, or video"}), 400

    saved_filename = None
    original_name  = None

    file = request.files.get("file")
    if file and file.filename:
        original_name = file.filename
        if not allowed_file(original_name, file_type):
            return jsonify({"success": False, "message": f"File type not allowed for {file_type}"}), 400
        safe_name      = wu.secure_filename(original_name)
        unique_name    = f"{uuid.uuid4().hex}_{safe_name}"
        dest           = os.path.join(UPLOAD_DIR, unique_name)
        file.save(dest)
        saved_filename = unique_name

    if file_type == "video" and not video_link:
        return jsonify({"success": False, "message": "Video link is required for video resources"}), 400

    resource = Resource(
        title         = title,
        description   = description,
        category      = category,
        keywords      = keywords,
        file_type     = file_type,
        filename      = saved_filename,
        original_name = original_name,
        video_link    = video_link if file_type == "video" else None,
        uploaded_by   = current_user.id,
    )
    db.session.add(resource)
    db.session.commit()
    return jsonify({"success": True, "message": "Resource uploaded", "resource": resource_to_dict(resource)}), 201


@app.route("/api/repository/<int:resource_id>", methods=["DELETE"])
@login_required
@require_role("admin", "superadmin")
def api_repo_delete(resource_id):
    """Delete a resource and optionally its uploaded file."""
    resource = db.session.get(Resource, resource_id)
    if not resource:
        return jsonify({"success": False, "message": "Resource not found"}), 404

    # Delete physical file if it exists
    if resource.filename:
        file_path = os.path.join(UPLOAD_DIR, resource.filename)
        if os.path.exists(file_path):
            os.remove(file_path)

    db.session.delete(resource)
    db.session.commit()
    return jsonify({"success": True, "message": "Resource deleted"})


@app.route("/api/repository/file/<path:filename>")
@login_required
def api_repo_serve_file(filename):
    """Serve an uploaded file."""
    from flask import send_from_directory
    return send_from_directory(UPLOAD_DIR, filename)



# Make login_required return JSON 401 instead of redirecting for API calls

@login_manager.unauthorized_handler
def unauthorized():
    if request.path.startswith("/api/") or request.is_json or request.headers.get("X-Requested-With") == "XMLHttpRequest":
        return jsonify({"success": False, "message": "Authentication required"}), 401
    return redirect(url_for("login"))

# =======================
# RUN
# =======================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)