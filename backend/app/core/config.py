from appwrite.client import Client
from appwrite.services.databases import Databases
from appwrite.services.storage import Storage
<<<<<<< HEAD
=======

>>>>>>> 926cbc3e2be1b9c532d2e65f6f7a731893a5a7ae
from dotenv import load_dotenv
import os

load_dotenv()


ENDPOINT = os.getenv("APPWRITE_ENDPOINT")
CLIENT = (
    Client()
    .set_endpoint(ENDPOINT)
    .set_project(os.environ["APPWRITE_PROJECT_ID"])
    .set_key(os.getenv("APPWRITE_API_KEY"))
    .set_self_signed(True)
)
PROJECT_ID = os.environ["APPWRITE_PROJECT_ID"]
<<<<<<< HEAD
=======
ENDPOINT = "http://localhost/v1"
>>>>>>> 926cbc3e2be1b9c532d2e65f6f7a731893a5a7ae
DATABASES = Databases(CLIENT)
DATABASE_ID = "agri_marketplace"
STORAGE = Storage(CLIENT)
# Collection IDs
COLLECTION_USERS = "users"
COLLECTION_CROP_LISTINGS = "crop_listings"
COLLECTION_BIDS = "bids"
COLLECTION_CONTRACTS = "contracts"
COLLECTION_CONTRACT_REQUESTS = "contract_requests"
COLLECTION_SUBSIDIES = "subsidies"
COLLECTION_SUBSIDY_REQUESTS = "subsidy_requests"
COLLECTION_ZIPCODES = "zipcodes"
<<<<<<< HEAD
=======
COLLECTION_CROP_PREDICTIONS = "crop_predictions"
>>>>>>> 926cbc3e2be1b9c532d2e65f6f7a731893a5a7ae
COLLECTION_DISEASE = "disease"
BUCKET_DISEASE = "disease"
COLLECTION_WEATHER = "weather"
COLLECTION_WEATHER_HISTORY = "weather_history"
BUCKET_SOIL = "soil"
COLLECTION_SOIL = "soil"
COLLECTION_PRICES = "prices"
COLLECTION_PRICES_HISTORY = "prices_history"
COLLECTION_CROPS = "crops"
BUCKET_CROP = "crop"
