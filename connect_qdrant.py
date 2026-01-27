from qdrant_client import QdrantClient

# Replace with your NEW details
client = QdrantClient(
    url="https://5248dcb1-2491-456b-8620-482380044a75.us-east4-0.gcp.cloud.qdrant.io",
    api_key="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.PGM-O0IyOg9ghiCNnS2atVZk7rYtoqv09SJS6MLRVtI"
)

# If this prints an empty list (or list of collections), you are connected!
print(client.get_collections())