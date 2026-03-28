"""One-time migration: add 'superadmin' to the users.role CHECK constraint."""
import os
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass
import psycopg2

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "pqnk_db"),
    user=os.getenv("DB_USER", "postgres"),
    password=os.getenv("DB_PASS", "admin123"),
    host=os.getenv("DB_HOST", "localhost"),
)
cur = conn.cursor()

# Drop old constraint (only includes seeker/farmer/researcher/admin)
cur.execute("ALTER TABLE users DROP CONSTRAINT IF EXISTS users_role_check")
# Add new constraint that also allows superadmin
cur.execute(
    "ALTER TABLE users ADD CONSTRAINT users_role_check "
    "CHECK (role IN ('seeker', 'farmer', 'researcher', 'admin', 'superadmin'))"
)
conn.commit()
print("OK: users_role_check updated to include superadmin")
cur.close()
conn.close()
