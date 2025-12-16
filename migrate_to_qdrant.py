import numpy as np
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import time
import logging

# Enable logging to see what's happening
logging.basicConfig(level=logging.INFO)

# --- CONFIG ---
QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
COLLECTION_NAME = "pqnk_vectors_v1"
VECTOR_DIM = 384

# Initialize client with timeout settings
client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60  # Increase timeout to 60 seconds
)

def test_connection():
    """Test connection to Qdrant"""
    try:
        collections = client.get_collections()
        print(f"✅ Connected to Qdrant! Available collections: {[col.name for col in collections.collections]}")
        return True
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def create_or_get_collection():
    """Create collection if it doesn't exist"""
    try:
        collections = client.get_collections().collections
        collection_names = [col.name for col in collections]
        
        if COLLECTION_NAME in collection_names:
            print(f"ℹ️ Collection '{COLLECTION_NAME}' already exists")
            return False  # Collection exists, no need to create
        else:
            print(f"🔄 Creating collection '{COLLECTION_NAME}'...")
            client.recreate_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
            )
            print(f"✅ Collection '{COLLECTION_NAME}' created")
            return True  # Collection was created
    except Exception as e:
        print(f"❌ Error creating collection: {e}")
        return False

def load_embeddings(lang):
    """Load embeddings from CSV and numpy files"""
    try:
        csv_file = f"embeddings_output/merged/{lang}_embeddings_merged.csv"
        npy_file = f"embeddings_output/merged/{lang}_vectors_merged.npy"
        
        print(f"📥 Loading {lang} data from {csv_file}...")
        df = pd.read_csv(csv_file, encoding='utf-8-sig')
        vectors = np.load(npy_file)
        
        print(f"📊 Loaded {len(df)} {lang} records, vectors shape: {vectors.shape}")
        return df, vectors
    except Exception as e:
        print(f"❌ Error loading {lang} embeddings: {e}")
        return None, None

def prepare_batch(df, vectors, start_idx, end_idx):
    """Prepare a batch of points from dataframe"""
    points = []
    for i in range(start_idx, end_idx):
        if i >= len(df):
            break
            
        row = df.iloc[i]
        point = PointStruct(
            id=i,  # Simple sequential ID
            vector=vectors[i].tolist(),
            payload={
                "text": str(row.get("text", "")),
                "filename": str(row.get("filename", "")),
                "category": str(row.get("category", "")),
                "language": str(row.get("language", "")),
                "chunk_id": int(row.get("chunk_id", 0)),
                "version": "v1"
            }
        )
        points.append(point)
    return points

def upload_with_retry(points, retries=3):
    """Upload points with retry logic"""
    for attempt in range(retries):
        try:
            client.upsert(
                collection_name=COLLECTION_NAME,
                points=points,
                wait=True  # Wait for confirmation
            )
            return True
        except Exception as e:
            if attempt < retries - 1:
                wait_time = 2 ** attempt  # Exponential backoff
                print(f"⚠️ Upload attempt {attempt + 1} failed: {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"❌ Upload failed after {retries} attempts: {e}")
                return False
    return False

def migrate_language(lang, batch_size=50):
    """Migrate embeddings for a single language"""
    print(f"\n{'='*50}")
    print(f"🚀 Migrating {lang.upper()} embeddings")
    print(f"{'='*50}")
    
    df, vectors = load_embeddings(lang)
    if df is None or vectors is None:
        print(f"❌ Skipping {lang} due to loading error")
        return 0
    
    total_points = len(df)
    print(f"📦 Total points to upload: {total_points}")
    
    successful_batches = 0
    failed_batches = 0
    
    for batch_start in range(0, total_points, batch_size):
        batch_end = min(batch_start + batch_size, total_points)
        batch_num = (batch_start // batch_size) + 1
        total_batches = (total_points + batch_size - 1) // batch_size
        
        print(f"\n📤 Batch {batch_num}/{total_batches}: Points {batch_start}-{batch_end}")
        
        # Prepare batch
        batch_points = prepare_batch(df, vectors, batch_start, batch_end)
        
        # Upload with retry
        if upload_with_retry(batch_points):
            successful_batches += 1
            print(f"✅ Batch {batch_num} uploaded successfully")
        else:
            failed_batches += 1
            print(f"❌ Batch {batch_num} failed")
        
        # Small delay between batches to avoid rate limiting
        if batch_end < total_points:
            time.sleep(0.5)
    
    print(f"\n📊 {lang.upper()} Migration Summary:")
    print(f"   Successful batches: {successful_batches}/{total_batches}")
    print(f"   Failed batches: {failed_batches}/{total_batches}")
    
    return successful_batches

def verify_migration():
    """Verify migration was successful"""
    try:
        # Import model for embedding
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Count points
        count_result = client.count(collection_name=COLLECTION_NAME)
        print(f"✅ Total points in collection: {count_result.count}")
        
        # Test search with a meaningful query
        test_vector = model.encode("agriculture farming").tolist()
        search_results = client.search(
            collection_name=COLLECTION_NAME,
            query_vector=test_vector,
            limit=2
        )
        
        if search_results:
            print("✅ Search test successful!")
            print(f"Found {len(search_results)} results")
            for i, result in enumerate(search_results):
                text_preview = result.payload.get('text', '')[:60]
                print(f"  Result {i+1}: Score={result.score:.3f}, Text='{text_preview}...'")
        else:
            print("⚠️ Search test returned no results")
            
    except Exception as e:
        print(f"⚠️ Verification failed: {e}")

def main():
    """Main migration function"""
    print("🚀 Starting Qdrant Migration")
    print(f"Target: {QDRANT_URL}")
    print(f"Collection: {COLLECTION_NAME}")
    print("="*50)
    
    # Test connection first
    if not test_connection():
        return
    
    # Create collection if needed
    collection_created = create_or_get_collection()
    
    if collection_created:
        print("🔄 New collection created, proceeding with migration...")
    else:
        # Check if collection is empty
        try:
            count_result = client.count(collection_name=COLLECTION_NAME)
            point_count = count_result.count
            print(f"ℹ️ Collection already has {point_count} points")
            
            if point_count > 0:
                response = input("⚠️ Collection already has data. Overwrite? (y/n): ")
                if response.lower() != 'y':
                    print("Migration cancelled.")
                    return
                else:
                    print("🔄 Overwriting existing data...")
                    client.delete_collection(collection_name=COLLECTION_NAME)
                    time.sleep(1)
                    create_or_get_collection()
        except Exception as e:
            print(f"⚠️ Could not check collection count: {e}")
    
    # Migrate both languages
    print("\n" + "="*50)
    print("🌍 Starting Migration Process")
    print("="*50)
    
    start_time = time.time()
    
    # Migrate English
    english_success = migrate_language("english", batch_size=30)  # Smaller batch size
    
    # Migrate Urdu
    urdu_success = migrate_language("urdu", batch_size=30)  # Smaller batch size
    
    end_time = time.time()
    
    print("\n" + "="*50)
    print("🎉 MIGRATION COMPLETE")
    print("="*50)
    print(f"Total time: {end_time - start_time:.2f} seconds")
    print(f"English batches: {english_success}")
    print(f"Urdu batches: {urdu_success}")
    
    # Verify migration
    verify_migration()

if __name__ == "__main__":
    main()