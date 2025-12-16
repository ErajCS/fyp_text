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





































import psycopg2
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, login_user, login_required, logout_user, UserMixin, current_user
from rag_demo import rag_pipeline
from datetime import datetime

# =======================
# FLASK APP CONFIG
# =======================
app = Flask(__name__)
app.config["SECRET_KEY"] = "supersecretkey"

# PostgreSQL connection URI (update username/password/dbname as needed)
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://postgres:admin123@localhost:5432/pqnk_db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

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
    phone = db.Column(db.String(20))  # new field
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
    error = None
    success = None

    if request.method == "POST":
        name = request.form["name"]
        email = request.form["email"]
        phone = request.form["phone"]
        password_raw = request.form["password"]
        confirm_password = request.form["confirm_password"]

        # Password match validation
        if password_raw != confirm_password:
            error = "Passwords do not match."
            return render_template("signup.html", error=error)

        # Check if email already exists
        if User.query.filter_by(email=email).first():
            error = "Email already exists."
            return render_template("signup.html", error=error)

        # Hash password
        password_hashed = bcrypt.generate_password_hash(password_raw).decode("utf-8")

        # Save new user
        new_user = User(
            name=name,
            email=email,
            phone=phone,
            password=password_hashed,
            role='seeker',        # default role
            dark_mode=False
        )

        db.session.add(new_user)
        db.session.commit()

        success = "Account created successfully! You may now login."
        return render_template("signup.html", success=success)

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
# CHATBOT ROUTES
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


# =======================
# RUN
# =======================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
