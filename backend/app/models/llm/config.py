# config.py
import os
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s')
log = logging.getLogger(__name__)

# --- File Paths ---
try:
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(PROJECT_ROOT, "data")
    AGRONOMIC_DB_PATH = os.path.join(DATA_DIR, "agronomic_database.json")
    CHROMA_DB_PATH = os.path.join(DATA_DIR, "chroma_news_db") # Directory for Chroma
    log.info(f"Data directory set to: {DATA_DIR}")
    log.info(f"ChromaDB persistent path: {CHROMA_DB_PATH}")
except Exception as e:
    log.error(f"Error setting up paths: {e}"); exit(1)

# --- Ollama Configuration ---
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma3:4b-it-q8_0") # Changed default model example
# --- MODIFIED: Provide default URL ---
OLLAMA_DEFAULT_URL = "http://localhost:11434"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", OLLAMA_DEFAULT_URL)
# --- END MODIFIED ---

LLM_TEMPERATURE = 0.1 # Default temperature for generation
LLM_TIMEOUT_SECONDS = 120 # Timeout for LLM requests
OLLAMA_TIMEOUT_SECONDS = LLM_TIMEOUT_SECONDS # Keep consistent naming if needed elsewhere

log.info(f"Using Ollama model: {OLLAMA_MODEL} at {OLLAMA_BASE_URL}")


# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
log.info(f"Using Embedding model: {EMBEDDING_MODEL_NAME}")

# --- Vector DB Config ---
CHROMA_COLLECTION_NAME = "agriculture_news"
NUM_RELEVANT_NEWS_TO_RETRIEVE = 5 # How many news items to fetch from ChromaDB for context

# --- Pipeline Parameters ---
MAX_RECOMMENDATIONS = 4
NEWS_MONTHS_TO_FETCH = 3 # Used by fetch_news_data.py for scraper lookback
HARD_CUTOFF_SUBSIDY_THRESHOLD = 0.8
ROUGH_INPUT_COST_ESTIMATES = {"Low": 10000, "Low-Medium": 15000, "Medium": 20000, "Medium-High": 25000, "High": 30000}


# --- Simulation Parameters ---
NUM_CROPS_TO_SIMULATE = 25 # Example if used

# --- Region Code Logic ---
def get_region_code(latitude: float, longitude: float) -> str:
    """Derives a simple region code based on latitude and longitude for India."""
    log.debug(f"Deriving region code for Latitude: {latitude}, Longitude: {longitude}")
    # Example: Simple grid for India (adjust ranges as needed)
    if 28.0 < latitude <= 33.0 and 73.0 < longitude <= 79.0: return "IN_NORTH" # Punjab, Haryana, HP, Uttarakhand
    if 23.0 < latitude <= 28.0 and 72.0 < longitude <= 80.0: return "IN_NORTH_CENTRAL" # Rajasthan, UP, North MP
    if 23.0 < latitude <= 28.0 and 80.0 < longitude <= 88.0: return "IN_EAST" # Bihar, Jharkhand, WB north
    if 20.0 < latitude <= 23.0 and 72.0 < longitude <= 80.0: return "IN_WEST_CENTRAL" # Gujarat, West MP, North Maharashtra
    if 20.0 < latitude <= 23.0 and 80.0 < longitude <= 88.0: return "IN_EAST_CENTRAL" # Odisha, Chhattisgarh, South WB
    if 15.0 < latitude <= 20.0 and 72.0 < longitude <= 80.0: return "IN_SOUTH_WEST" # Maharashtra, Karnataka, Goa
    if 10.0 < latitude <= 15.0 and 75.0 < longitude <= 80.0: return "IN_SOUTH" # Kerala, Tamil Nadu, South AP/Telangana
    # Fallback / Default
    return "IN_OTHER"

# --- Scraper Configuration ---
SEARCH_QUERIES = [ "agriculture India", "crop prices India", "monsoon forecast India", "fertilizer subsidy India", "pest outbreak India crops", "farmer protest India", "MSP crop India", "agriculture export import India", "agritech India", "farm loan India", "weather agriculture India", "water irrigation India" ]
GOOGLE_NEWS_URL = "https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
REQUEST_DELAY_SECONDS = 1.5
SCRAPER_TIMEOUT_SECONDS = 25