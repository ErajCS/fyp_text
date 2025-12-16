import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import os # Import the os module to help build file paths

# --- 1. Configuration ---

# This MUST be the same model you used to create the embeddings
MODEL_NAME = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'

# Define the base directory for your indexes
BASE_INDEX_DIR = "faiss_indexes" 

# Ask the user which language to search
lang = input("Which language to search? (en/ur): ").strip().lower()

if lang == 'ur':
    # --- UPDATED FILE PATHS ---
    FAISS_INDEX_FILE = os.path.join(BASE_INDEX_DIR, "urdu_fiass.index")
    METADATA_FILE = os.path.join(BASE_INDEX_DIR, "urdu_metadata.json")
    print("Loading URDU database...")
elif lang == 'en':
    # --- UPDATED FILE PATHS ---
    # Note: Your filename is 'englis_metadata.json', I've corrected it here.
    # If 'englis_metadata.json' is the real name, change it below.
    FAISS_INDEX_FILE = os.path.join(BASE_INDEX_DIR, "english_faiss.index")
    METADATA_FILE = os.path.join(BASE_INDEX_DIR, "english_metadata.json") # Corrected from 'englis'
    print("Loading ENGLISH database...")
else:
    print("Invalid language. Exiting.")
    exit()

# --- 2. Load Model and Data ---

try:
    # Load the FAISS index
    index = faiss.read_index(FAISS_INDEX_FILE)

    # Load the metadata
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        metadata_list = json.load(f)
        
    # Convert metadata list to a dictionary for fast lookups
    metadata_map = {item['id']: item for item in metadata_list}

    # Load the Sentence Transformer model
    print("Loading embedding model...")
    model = SentenceTransformer(MODEL_NAME)
    print("✅ Database and model loaded successfully!")

except FileNotFoundError:
    print(f"Error: Could not find '{FAISS_INDEX_FILE}' or '{METADATA_FILE}'.")
    print(f"Please check your directory structure. Looking in: {os.path.abspath(BASE_INDEX_DIR)}")
    exit()
except Exception as e:
    print(f"An error occurred during loading: {e}")
    exit()

# --- 3. The Search Function ---

def search(query_text, k=3):
    """
    Performs a semantic search.
    
    1. Converts the query_text to an embedding.
    2. Searches the FAISS index for the k-nearest neighbors.
    3. Looks up the metadata for those neighbors and returns them.
    """
    if not query_text:
        return []

    print(f"\nEmbedding query: '{query_text}'")
    
    # 1. Convert the query text to an embedding (vector)
    query_vector = model.encode(query_text)
    
    # 2. FAISS requires a 2D numpy array, so we reshape and ensure float32
    query_vector_np = np.array([query_vector]).astype('float32')

    # 3. Search the FAISS index
    # D = distances (how far), I = indices (the 'id's from your metadata)
    try:
        D, I = index.search(query_vector_np, k)
    except Exception as e:
        print(f"Error during FAISS search: {e}")
        return []

    # 4. Use the indices (I) to look up the metadata
    results = []
    
    # I is a 2D array (e.g., [[504, 1, 0]]), so we get the first list
    indices = I[0] 
    
    print(f"Found {len(indices)} matching chunks (IDs: {indices})...")
    
    for idx in indices:
        # Use our fast metadata_map to get the result
        # .item() converts numpy int to native python int for JSON key
        result = metadata_map.get(idx.item()) 
        if result:
            results.append(result)
        else:
            print(f"Warning: Could not find metadata for ID {idx.item()}")
            
    return results

# --- 4. Main Program Loop ---

if __name__ == "__main__":
    # Place this script in your main FYP_TEXT directory
    # It will look for the 'faiss_indexes' folder relative to itself.
    print(f"FYP_TEXT retrieval script running from: {os.getcwd()}")
    
    print("\n--- PQNK Semantic Search ---")
    print("Type your query and press Enter. Type 'q' to quit.")
    
    while True:
        query = input("\nQuery: ")
        
        if query.lower() == 'q':
            break
            
        # 5. Perform the search
        search_results = search(query, k=3)
        
        # 6. Print the results
        if not search_results:
            print("No results found.")
            continue
            
        print("\n--- Top Results ---")
        for i, res in enumerate(search_results):
            print(f"\nResult {i+1}:")
            print(f"  Source: {res['filename']}")
            print(f"  Category: {res['category']}")
            print(f"  Text: ...{res['text'][:500]}...") # Print first 500 chars