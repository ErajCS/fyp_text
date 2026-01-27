
from qdrant_client import QdrantClient
import os

client = QdrantClient(
   QDRANT_API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM",
   QDRANT_URL = "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io"
# COLLECTION_NAME = "pqnk_vectors_v1"
# OPENAI_API_KEY = "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"

)

print(client.get_collections())
