import psycopg2
import os
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


DB_USER = os.getenv("DB_USER", "postgres")
DB_PASS = os.getenv("DB_PASS", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME", "pqnk_db")



def create_database():
    conn = psycopg2.connect(
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST,
        dbname="postgres"
    )
    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    cur.execute(f"SELECT 1 FROM pg_database WHERE datname = '{DB_NAME}'")
    if not cur.fetchone():
        print("Creating database...")
        cur.execute(f"CREATE DATABASE {DB_NAME}")
    else:
        print("Database already exists.")

    cur.close()
    conn.close()


def create_tables():
    conn = psycopg2.connect(
        database=DB_NAME,
        user=DB_USER,
        password=DB_PASS,
        host=DB_HOST
    )
    register_vector(conn)
    cur = conn.cursor()

    print("Enabling extensions...")
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm;")

    print("Creating tables...")

    print("Creating users table...")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id SERIAL PRIMARY KEY,
            full_name VARCHAR(100) NOT NULL,
            email VARCHAR(255) UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role VARCHAR(20) CHECK (role IN ('admin', 'seeker')),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS content (
            content_id SERIAL PRIMARY KEY,
            filename VARCHAR(255),
            category VARCHAR(100),
            language VARCHAR(10),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS vectors (
            vector_id SERIAL PRIMARY KEY,
            content_id INT REFERENCES content(content_id) ON DELETE CASCADE,
            chunk_text TEXT,
            chunk_index INT,
            embedding vector(384),
            text_search_tokens tsvector
        );            
    """)

    print("Ensuring tsvector column exists...")
    cur.execute("""
        ALTER TABLE vectors
        ADD COLUMN IF NOT EXISTS text_search_tokens tsvector;
    """)

    print("Backfilling tsvector...")
    cur.execute("""
        UPDATE vectors
        SET text_search_tokens = to_tsvector('english', chunk_text)
        WHERE text_search_tokens IS NULL;
    """)

    print("Creating indexes...")
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_vectors_embedding
        ON vectors USING ivfflat (embedding vector_cosine_ops);
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_vectors_tsvector
        ON vectors USING GIN (text_search_tokens);
    """)

    cur.execute("""
        CREATE OR REPLACE FUNCTION update_tsvector_column()
        RETURNS trigger AS $$
        BEGIN
          NEW.text_search_tokens := to_tsvector('english', NEW.chunk_text);
          RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """)

    cur.execute("""
        DROP TRIGGER IF EXISTS tsvector_update ON vectors;
        CREATE TRIGGER tsvector_update
        BEFORE INSERT OR UPDATE ON vectors
        FOR EACH ROW EXECUTE FUNCTION update_tsvector_column();
    """)

    conn.commit()
    cur.close()
    conn.close()

    print("✅ Database fully hybrid-search ready.")


if __name__ == "__main__":
    create_database()
    create_tables()
