import psycopg2
import bcrypt

# --- CONFIG ---
DB_NAME = "pqnk_db"
DB_USER = "postgres"
DB_PASS = "admin123"

def seed_users():
    conn = psycopg2.connect(dbname=DB_NAME, user=DB_USER, password=DB_PASS, host="localhost")
    cur = conn.cursor()

    users = [
        ("Dr. Asif Sharif", "admin@pqnk.com", "admin123", "admin"),
        ("Ali Farmer", "ali@farmer.com", "farmer123", "seeker"),
        ("Sara Student", "sara@uni.edu", "student123", "seeker")
    ]

    print("🌱 Seeding Users...")
    for name, email, password, role in users:
        # Hash the password
        hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        
        try:
            cur.execute("""
                INSERT INTO users (full_name, email, password_hash, role)
                VALUES (%s, %s, %s, %s)
            """, (name, email, hashed, role))
            print(f"✅ Added {role}: {email}")
        except psycopg2.errors.UniqueViolation:
            print(f"⚠️ User {email} already exists. Skipping.")
            conn.rollback()
        else:
            conn.commit()

    cur.close()
    conn.close()

if __name__ == "__main__":
    seed_users()