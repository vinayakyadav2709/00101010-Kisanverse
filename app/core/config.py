from appwrite.client import Client
from appwrite.services.databases import Databases

from dotenv import load_dotenv
import os

load_dotenv()
CLIENT = (
    Client()
    .set_endpoint("http://localhost/v1")
    .set_project(os.environ["APPWRITE_PROJECT_ID"])
    .set_key(os.getenv("APPWRITE_API_KEY"))
    .set_self_signed(True)
)

DATABASES = Databases(CLIENT)
DATABASE_ID = "agri_marketplace"

# Collection IDs
COLLECTION_USERS = "users"
COLLECTION_CROP_LISTINGS = "crop_listings"
COLLECTION_BIDS = "bids"
COLLECTION_CONTRACTS = "contracts"
COLLECTION_CONTRACT_REQUESTS = "contract_requests"
COLLECTION_SUBSIDIES = "subsidies"
COLLECTION_SUBSIDY_REQUESTS = "subsidy_requests"
