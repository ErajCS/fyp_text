# # from qdrant_client import QdrantClient

# # # Try both possible URL formats
# # urls_to_test = [
# #     "https://d002b9a6-4c34-4bc3-b26d-9b3bdbcc6b4b.us-east4-0.gcp.cloud.qdrant.io",  # Your actual URL
# #     "https://pqnk.qdrant.cloud"  # This likely doesn't exist
# # ]

# # API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.4-OvJVbv3S1tGqBDLMOZCzt_VKVT24Ac2Za2_N5wtBM"

# # for url in urls_to_test:
# #     print(f"\nTesting URL: {url}")
# #     try:
# #         client = QdrantClient(url=url, api_key=API_KEY, timeout=10)
# #         collections = client.get_collections()
# #         print(f"✅ SUCCESS! Collections: {[col.name for col in collections.collections]}")
# #         break  # Stop at first successful URL
# #     except Exception as e:
# #         print(f"❌ FAILED: {e}")







# import qdrant_client
# import pkg_resources

# # Check version
# try:
#     version = pkg_resources.get_distribution("qdrant-client").version
#     print(f"Qdrant Client Version: {version}")
# except:
#     print("Could not get version via pkg_resources")

# # Alternative way
# print(f"Qdrant module location: {qdrant_client.__file__}")

# # List available attributes
# print("\nAvailable attributes in qdrant_client module:")
# for attr in dir(qdrant_client):
#     if not attr.startswith('_'):
#         print(f"  {attr}")

















from qdrant_client import QdrantClient

client = QdrantClient(":memory:")

methods = [m for m in dir(client) if not m.startswith("_")]

print("Available methods in QdrantClient:\n")
for m in methods:
    print(m)
