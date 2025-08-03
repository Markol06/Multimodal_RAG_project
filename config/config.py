import os

from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("No OPENAI_API_KEY found. Please check your .env file.")

TEXT_EMBEDDING_MODEL = "text-embedding-3-small"
IMAGE_EMBEDDING_MODEL = "openai/clip-vit-large-patch14"
CHAT_MODEL = "gpt-4o"
TEMPERATURE_STRICT = 0.2
TEMPERATURE_CREATIVE = 0.7

TOP_K = 5

BASE_URL = "https://www.deeplearning.ai"
START_URL = f"{BASE_URL}/the-batch/"
NUM_ARTICLES = 10

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
