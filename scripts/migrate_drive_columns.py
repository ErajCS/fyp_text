"""
migrate_drive_columns.py
Add drive_file_id and drive_view_link columns to the resources table.
Run once: python migrate_drive_columns.py
"""
import os
try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env'))
except ImportError:
    pass
import psycopg2

conn = psycopg2.connect(
    dbname   = os.getenv("DB_NAME", "pqnk_db"),
    user     = os.getenv("DB_USER", "postgres"),
    password = os.getenv("DB_PASS", "admin123"),
    host     = os.getenv("DB_HOST", "localhost"),
)
cur = conn.cursor()

cur.execute("""
    ALTER TABLE resources
    ADD COLUMN IF NOT EXISTS drive_file_id    VARCHAR(100),
    ADD COLUMN IF NOT EXISTS drive_view_link  VARCHAR(512);
""")
conn.commit()
print("OK: drive_file_id and drive_view_link columns added to resources table")
cur.close()
conn.close()
