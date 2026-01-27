
from qdrant_client import QdrantClient
import os

client = QdrantClient(
   QDRANT_API_KEY = os.getenv("QDRANT_API_KEY"),
# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
   QDRANT_URL = os.getenv("QDRANT_URL")  
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

)

print(client.get_collections())
