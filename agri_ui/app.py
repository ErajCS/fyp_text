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

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# =======================
# HELPER: TEXT TO SPEECH
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
    user_message = request.json.get("msg")
    ai_response = rag_pipeline(user_message)
    return jsonify({"response": ai_response})

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