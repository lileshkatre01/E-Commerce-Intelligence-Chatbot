import os
from pathlib import Path
from dotenv import load_dotenv

# Base project directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load environment variables from .env if present
load_dotenv(BASE_DIR / ".env")

# API Keys & Model configurations
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

# Embedding & Search configurations
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))
CANDIDATE_POOL_MULTIPLIER = int(os.getenv("CANDIDATE_POOL_MULTIPLIER", "3"))

# Data and Index Paths
DATA_DIR = BASE_DIR / "data"
REVIEWS_CSV_PATH = DATA_DIR / "reviews.csv"
FAISS_INDEX_DIR = DATA_DIR / "faiss_index"
FAISS_INDEX_PATH = FAISS_INDEX_DIR / "reviews.index"
DOCUMENTS_PICKLE_PATH = FAISS_INDEX_DIR / "documents.pkl"

# Dataset fallback source URL
DATASET_URL = "https://raw.githubusercontent.com/DataScience75/Top_mentor_projects_Datasets/refs/heads/main/customer_reviews.csv"
