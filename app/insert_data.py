from appwrite.id import ID
from core.config import (
    DATABASE_ID,
    COLLECTION_USERS,
    COLLECTION_CROP_LISTINGS,
    COLLECTION_BIDS,
    COLLECTION_CONTRACTS,
    COLLECTION_CONTRACT_REQUESTS,
    COLLECTION_SUBSIDIES,
    COLLECTION_SUBSIDY_REQUESTS,
    DATABASES,
)
from datetime import datetime, timedelta, timezone


def clear_collection(collection_id):
    """
    Deletes all documents in the specified collection.
    """
    print(f"Clearing collection: {collection_id}")
    documents = DATABASES.list_documents(
        database_id=DATABASE_ID, collection_id=collection_id
    )["documents"]
    for document in documents:
        DATABASES.delete_document(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            document_id=document["$id"],
        )
    print(f"Cleared all documents in collection: {collection_id}")


# Clear all collections before inserting new data
collections_to_clear = [
    COLLECTION_USERS,
    COLLECTION_CROP_LISTINGS,
    COLLECTION_BIDS,
    COLLECTION_CONTRACTS,
    COLLECTION_CONTRACT_REQUESTS,
    COLLECTION_SUBSIDIES,
    COLLECTION_SUBSIDY_REQUESTS,
]

for collection in collections_to_clear:
    clear_collection(collection)


def get_date(days_offset):
    """
    Given an integer, returns a datetime object that many days before or after the current date.
    Positive integer -> Future date
    Negative integer -> Past date
    """
    current_datetime = datetime.now(timezone.utc)
    target_datetime = current_datetime + timedelta(days=days_offset)
    return target_datetime.isoformat()


# Users
users = [
    {
        "email": "farmer@example.com",
        "name": "Ramesh Sahu",
        "role": "farmer",
        "zipcode": "493887",
        "address": "Myfarm Naturals, Nandanvan main road, chandandih, Raipur, Chandanidih, CHHATTISGARH",
    },
    {
        "email": "admin@example.com",
        "name": "Sangeeta Sharma",
        "role": "admin",
        "zipcode": "000000",
        "address": "The/Nudge HQ, Bengaluru, Karnataka",
    },
    {
        "email": "demobuyer1@example.com",
        "name": "Ankit Verma",
        "role": "buyer",
        "zipcode": "123456",
        "address": "Plot 23, M.G. Road, Raipur, CHHATTISGARH",
    },
    {
        "email": "demobuyer2@example.com",
        "name": "Priya Singh",
        "role": "buyer",
        "zipcode": "654321",
        "address": "Flat 502, Green Residency, Bilaspur, CHHATTISGARH",
    },
    {
        "email": "demobuyer3@example.com",
        "name": "Rajesh Patel",
        "role": "buyer",
        "zipcode": "789012",
        "address": "15 Nehru Nagar, Durg, CHHATTISGARH",
    },
    {
        "email": "demobuyer4@example.com",
        "name": "Kavita Nair",
        "role": "buyer",
        "zipcode": "210987",
        "address": "302, Sunshine Apartments, Korba, CHHATTISGARH",
    },
    {
        "email": "demobuyer5@example.com",
        "name": "Siddharth Iyer",
        "role": "buyer",
        "zipcode": "345678",
        "address": "B-14, Ravi Enclave, Jagdalpur, CHHATTISGARH",
    },
]

user_ids = []
for user in users:
    response = DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_USERS,
        document_id=ID.unique(),
        data=user,
    )
    user_ids.append(response["$id"])

# Crop listing by farmer
listing = {
    "farmer_id": user_ids[0],
    "crop_type": "WHEAT",
    "price_per_kg": 20.0,
    "total_quantity": 100.0,
    "available_quantity": 100.0,
    "status": "listed",
    "added_at": get_date(10),
    "updated_at": get_date(10),
}
listing_response = DATABASES.create_document(
    database_id=DATABASE_ID,
    collection_id=COLLECTION_CROP_LISTINGS,
    document_id=ID.unique(),
    data=listing,
)
listing_id = listing_response["$id"]

# Bids from buyers
bids = [
    {
        "buyer_id": user_ids[2],
        "listing_id": listing_id,
        "quantity": 10.0,
        "price_per_kg": 18.0,
        "status": "rejected",
        "timestamp": get_date(-5),
    },
    {
        "buyer_id": user_ids[3],
        "listing_id": listing_id,
        "quantity": 20.0,
        "price_per_kg": 20.0,
        "status": "accepted",
        "timestamp": get_date(-5),
    },
    {
        "buyer_id": user_ids[4],
        "listing_id": listing_id,
        "quantity": 15.0,
        "price_per_kg": 21.0,
        "status": "accepted",
        "timestamp": get_date(-5),
    },
    {
        "buyer_id": user_ids[5],
        "listing_id": listing_id,
        "quantity": 25.0,
        "price_per_kg": 22.0,
        "status": "pending",
        "timestamp": get_date(-4),
    },
    {
        "buyer_id": user_ids[6],
        "listing_id": listing_id,
        "quantity": 30.0,
        "price_per_kg": 23.0,
        "status": "fulfilled",
        "timestamp": get_date(-2),
    },
]

bid_ids = []
total_sold = 0
for bid in bids:
    response = DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_BIDS,
        document_id=ID.unique(),
        data=bid,
    )
    bid_ids.append(response["$id"])
    if bid["status"] in ["accepted", "fulfilled"]:
        total_sold += bid["quantity"]

# Update available quantity
remaining_quantity = listing["total_quantity"] - total_sold
DATABASES.update_document(
    database_id=DATABASE_ID,
    collection_id=COLLECTION_CROP_LISTINGS,
    document_id=listing_id,
    data={"available_quantity": remaining_quantity},
)

import json

contracts = [
    {
        "buyer_id": user_ids[2],
        "crop_type": "RICE",
        "quantity": 1000,
        "price_per_kg": 25.0,
        "locations": ["CHHATTISGARH", "Delhi"],
        "advance_payment": 5000,
        "delivery_date": get_date(180),
        "payment_terms": "50% advance, 50% on delivery",
        "status": "accepted",
        "dynamic_fields": json.dumps(
            {"requirements": {"grade_required": "A", "transport_required": True}}
        ),
    },
    {
        "buyer_id": user_ids[3],
        "crop_type": "MAIZE",
        "quantity": 800,
        "price_per_kg": 22.0,
        "locations": ["CHHATTISGARH", "Maharashtra"],
        "advance_payment": 4000,
        "delivery_date": get_date(180),
        "payment_terms": "Cash on delivery",
        "status": "fulfilled",
        "dynamic_fields": json.dumps(
            {
                "requirements": {
                    "certification": "organic",
                    "storage_condition": "cool dry place",
                }
            }
        ),
    },
    {
        "buyer_id": user_ids[4],
        "crop_type": "MANGO",
        "quantity": 600,
        "price_per_kg": 30.0,
        "locations": ["CHHATTISGARH", "West Bengal"],
        "advance_payment": 3000,
        "delivery_date": get_date(150),
        "payment_terms": "Full payment on delivery",
        "status": "listed",
        "dynamic_fields": json.dumps(
            {
                "requirements": {
                    "harvest_window": "Oct-Nov",
                    "packaging_standard": "standard boxes",
                }
            }
        ),
    },
    {
        "buyer_id": user_ids[5],
        "crop_type": "ONION",
        "quantity": 500,
        "price_per_kg": 18.0,
        "locations": ["CHHATTISGARH", "Gujarat"],
        "advance_payment": 2000,
        "delivery_date": get_date(185),
        "payment_terms": "Advance payment only",
        "status": "listed",
        "dynamic_fields": json.dumps(
            {"requirements": {"bulk_order_only": True, "min_diameter_mm": 40}}
        ),
    },
    {
        "buyer_id": user_ids[6],
        "crop_type": "POTATO",
        "quantity": 700,
        "price_per_kg": 20.0,
        "locations": ["CHHATTISGARH", "Karnataka"],
        "advance_payment": 3500,
        "delivery_date": get_date(170),
        "payment_terms": "50% advance, 50% on delivery",
        "status": "listed",
        "dynamic_fields": json.dumps(
            {"requirements": {"refrigeration_required": True, "grade_required": "B"}}
        ),
    },
]

contract_ids = []
for contract in contracts:
    response = DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_CONTRACTS,
        document_id=ID.unique(),
        data=contract,
    )
    contract_ids.append(response["$id"])

# Contract requests
requests = [
    {"contract_id": contract_ids[0], "farmer_id": user_ids[0], "status": "accepted"},
    {"contract_id": contract_ids[1], "farmer_id": user_ids[0], "status": "fulfilled"},
    {"contract_id": contract_ids[2], "farmer_id": user_ids[0], "status": "pending"},
]
for request in requests:
    DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_CONTRACT_REQUESTS,
        document_id=ID.unique(),
        data=request,
    )

# Subsidies from The/Nudge Institute
subsidies = [
    {
        "program": "Green Fields",
        "description": "Eco-friendly seed support",
        "eligibility": "Small and marginal farmers practicing sustainable farming in rural CHHATTISGARH",
        "type": "cash",
        "benefits": "INR 5000",
        "application_process": "Submit online application with land ownership proof",
        "locations": ["CHHATTISGARH", "Karnataka"],
        "dynamic_fields": json.dumps(
            {
                "seed_type": "eco",
                "proof_required": True,
                "contact_info": "support@nudge.org.in",
                "application_link": "https://thenudge.org/green-fields-apply",
            }
        ),
        "max_recipients": 50,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
    {
        "program": "Water Saver",
        "description": "Drip irrigation support for dryland farmers",
        "eligibility": "Registered farmers with less than 2 hectares under cultivation",
        "type": "asset",
        "benefits": "Free drip irrigation kit (upto 2 acres)",
        "application_process": "Upload land records and irrigation need report",
        "locations": ["CHHATTISGARH", "Madhya Pradesh"],
        "dynamic_fields": json.dumps(
            {
                "equipment_type": "drip",
                "max_area_covered": "2 acres",
                "contact_info": "+91-800-456-7890",
                "application_link": "https://thenudge.org/water-saver",
            }
        ),
        "max_recipients": 30,
        "provider": "The/Nudge Institute",
        "status": "listed",
        "recipients_accepted": 1,
    },
    {
        "program": "Organic Boost",
        "description": "Incentive for certified organic farming",
        "eligibility": "Farmers with government-recognized organic certification",
        "type": "cash",
        "benefits": "INR 10,000",
        "application_process": "Apply via The/Nudge mobile app with certificate",
        "locations": ["CHHATTISGARH", "Odisha"],
        "dynamic_fields": json.dumps(
            {
                "requires_certification": True,
                "contact_info": "orgsupport@nudge.org.in",
                "application_link": "https://thenudge.org/organic-boost",
            }
        ),
        "max_recipients": 20,
        "provider": "The/Nudge Institute",
        "status": "listed",
        "recipients_accepted": 1,
    },
    {
        "program": "Soil Health Mission",
        "description": "Support for soil testing and nutrient advisory",
        "eligibility": "All farmers with valid land documents and Aadhaar verification",
        "type": "training",
        "benefits": "Free soil testing and seasonal crop advisory",
        "application_process": "Visit nearest Krishi Kendra or apply via official website",
        "locations": ["CHHATTISGARH", "Uttar Pradesh"],
        "dynamic_fields": json.dumps(
            {
                "service_type": "testing",
                "turnaround_time": "7 days",
                "contact_info": "soilhealth@nudge.org.in",
                "application_link": "https://thenudge.org/soil-health",
            }
        ),
        "max_recipients": 100,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
    {
        "program": "AgriTech Connect",
        "description": "Digital toolkits and training for smallholder farmers",
        "eligibility": "Young or women farmers interested in using agri-tech tools",
        "type": "asset",
        "benefits": "Free smartphone + 3-month training access",
        "application_process": "Nomination via SHG or FPO partners",
        "locations": ["CHHATTISGARH"],
        "dynamic_fields": json.dumps(
            {
                "includes_device": True,
                "duration": "3 months",
                "contact_info": "training@nudge.org.in",
                "application_link": "https://thenudge.org/agritech-connect",
            }
        ),
        "max_recipients": 40,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
    {
        "program": "Farm Mechanization Support",
        "description": "Subsidy for renting or purchasing farm machinery",
        "eligibility": "Smallholder farmers with <5 acres landholding",
        "type": "cash",
        "benefits": "Upto 50% subsidy on machinery rentals or purchase",
        "application_process": "Apply online with landholding certificate and quotation from supplier",
        "locations": ["CHHATTISGARH", "Telangana"],
        "dynamic_fields": json.dumps(
            {
                "machinery_type": "tractor/power tiller",
                "subsidy_limit": "INR 25,000",
                "contact_info": "mechanize@nudge.org.in",
                "application_link": "https://thenudge.org/mechanization",
            }
        ),
        "max_recipients": 35,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
    {
        "program": "Women in Farming",
        "description": "Empowerment grant for women-led farms and SHGs",
        "eligibility": "Women farmers or SHG leaders registered in any rural CHHATTISGARH panchayat",
        "type": "cash",
        "benefits": "INR 7,500 + optional training",
        "application_process": "Apply via SHG recommendation letter and ID proof",
        "locations": ["CHHATTISGARH", "Jharkhand"],
        "dynamic_fields": json.dumps(
            {
                "requires_shg_membership": True,
                "training_included": True,
                "contact_info": "womenfarming@nudge.org.in",
                "application_link": "https://thenudge.org/women-farm",
            }
        ),
        "max_recipients": 60,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
    {
        "program": "Climate Resilience Kit",
        "description": "Kit of tools and inputs for climate-resilient farming",
        "eligibility": "Farmers in flood/drought-prone blocks of CHHATTISGARH",
        "type": "asset",
        "benefits": "Resilient seeds, fertilizers, weather training",
        "application_process": "Block-level application through agriculture officer",
        "locations": ["CHHATTISGARH", "Bihar"],
        "dynamic_fields": json.dumps(
            {
                "includes_training": True,
                "kit_components": [
                    "resilient seeds",
                    "bio-fertilizer",
                    "weather booklet",
                ],
                "contact_info": "climate@nudge.org.in",
                "application_link": "https://thenudge.org/climate-kit",
            }
        ),
        "max_recipients": 45,
        "provider": "The/Nudge Institute",
        "status": "listed",
    },
]


subsidy_ids = []
for subsidy in subsidies:
    response = DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_SUBSIDIES,
        document_id=ID.unique(),
        data=subsidy,
    )
    subsidy_ids.append(response["$id"])

# Subsidy requests by the farmer
subsidy_requests = [
    {"subsidy_id": subsidy_ids[0], "farmer_id": user_ids[0], "status": "requested"},
    {"subsidy_id": subsidy_ids[1], "farmer_id": user_ids[0], "status": "accepted"},
    {"subsidy_id": subsidy_ids[2], "farmer_id": user_ids[0], "status": "fulfilled"},
]
for request in subsidy_requests:
    DATABASES.create_document(
        database_id=DATABASE_ID,
        collection_id=COLLECTION_SUBSIDY_REQUESTS,
        document_id=ID.unique(),
        data=request,
    )

print("🎉 Demo data for CHHATTISGARH farmer ecosystem inserted successfully!")
