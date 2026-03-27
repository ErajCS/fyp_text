import os
import psycopg2
import bcrypt
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# --- CONFIG (from .env) ---
DB_NAME = os.getenv("DB_NAME", "pqnk_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")


def seed_users():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
    cur = conn.cursor()

    users = [
        ("Dr. Asif Sharif", "admin@pqnk.com", "admin123", "admin"),
        ("System Owner", "superadmin@pqnk.com", "super123", "superadmin"),
        ("Ali Farmer", "ali@farmer.com", "farmer123", "farmer"),
        ("Sara Student", "sara@uni.edu", "student123", "seeker")
    ]


    print("🌱 Seeding Users...")
    for name, email, password, role in users:
        # Hash the password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            cur.execute("""
                INSERT INTO users (full_name, email, password_hash, role, is_verified)
                VALUES (%s, %s, %s, %s, TRUE)
                ON CONFLICT (email) DO UPDATE
                  SET full_name=EXCLUDED.full_name, role=EXCLUDED.role, is_verified=TRUE
            """, (name, email, hashed, role))
            print(f"✅ Upserted {role}: {email}")

        except psycopg2.errors.UniqueViolation:
            print(f"⚠️ User {email} already exists. Skipping.")
            conn.rollback()
        else:
            conn.commit()

    cur.close()
    conn.close()

if __name__ == "__main__":
    seed_users()