# import psycopg2
# from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
# from pgvector.psycopg2 import register_vector
# import os

# # --- CONFIGURATION (Ensure this matches your installation) ---
# DB_USER = "postgres"
# DB_PASS = "admin123"  # <--- CONFIRM/CHANGE THIS TO YOUR PASSWORD
# DB_HOST = "localhost"
# DB_NAME = "pqnk_db"

# def create_database():
#     """Connects to default 'postgres' db to create the target database."""
#     try:
#         conn = psycopg2.connect(user=DB_USER, password=DB_PASS, host=DB_HOST, dbname="postgres")
#         conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
#         cur = conn.cursor()
#     except psycopg2.OperationalError as e:
#         print(f"❌ DATABASE CONNECTION ERROR: {e}")
#         print("Ensure PostgreSQL service is running and the USER/PASS are correct.")
#         return False
    
#     cur.execute(f"SELECT 1 FROM pg_catalog.pg_database WHERE datname = '{DB_NAME}'")
#     if not cur.fetchone():
#         print(f"Creating database {DB_NAME}...")
#         cur.execute(f"CREATE DATABASE {DB_NAME}")
#     else:
#         print(f"Database {DB_NAME} already exists.")
    
#     cur.close()
#     conn.close()
#     return True

# def create_tables():
#     """Connects to PQNK DB and creates tables/extension."""
#     try:
#         conn = psycopg2.connect(database=DB_NAME, user=DB_USER, password=DB_PASS, host=DB_HOST)
#         # Register the vector type handle
#         register_vector(conn) 
#         cur = conn.cursor()
#     except psycopg2.OperationalError as e:
#         print(f"❌ DATABASE CONNECTION ERROR: {e}. Was the database created in the previous step?")
#         return
    
#     print("Enabling pgvector extension...")
#     # This MUST be successful now that the pgvector files are in place.
#     cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    
#     print("Creating tables (if not exist)...")
    
#     # 1. Content Table (Metadata about the file)
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS content (
#             content_id SERIAL PRIMARY KEY,
#             filename VARCHAR(255),
#             category VARCHAR(100),
#             language VARCHAR(10),
#             created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
#         );
#     """)

#     # 2. Vectors Table (The embeddings and text chunks - Dimension 384 for MiniLM)
#     cur.execute("""
#         CREATE TABLE IF NOT EXISTS vectors (
#             vector_id SERIAL PRIMARY KEY,
#             content_id INT REFERENCES content(content_id) ON DELETE CASCADE,
#             chunk_text TEXT,
#             chunk_index INT,
#             embedding vector(384),
#             -- NEW COLUMN: Stores tokens for fast keyword search (BM25)
#             text_search_tokens tsvector 
#         );
#     """)
    

#     # 3. Create the GIN Index for fast full-text searching
#     cur.execute("CREATE INDEX IF NOT EXISTS idx_vectors_tsvector ON vectors USING GIN (text_search_tokens);")
#     conn.commit()
#     cur.close()
#     conn.close()
#     print("✅ Database and Tables setup complete!")

# if __name__ == "__main__":
#     if create_database():
#         create_tables()






















import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from pgvector.psycopg2 import register_vector

DB_USER = "postgres"
DB_PASS = "admin123"
DB_HOST = "localhost"
DB_NAME = "pqnk_db"


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
