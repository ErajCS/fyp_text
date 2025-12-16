
from qdrant_client import QdrantClient

client = QdrantClient(
    url="https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"
)

print(client.get_collections())
