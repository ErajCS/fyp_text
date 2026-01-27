from dotenv import load_dotenv
import os

load_dotenv()  # This loads variables from .env into the OS environment

api_key = os.getenv("API_KEY")

print(api_key)  # just to test
