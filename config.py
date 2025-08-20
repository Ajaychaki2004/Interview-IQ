import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Google API Key - Single key for the entire application
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
