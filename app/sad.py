import json
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
from appwrite.query import Query


def fetch_all_data():
    """
    Fetch all data from the specified collections and store it in JSON format.
    The format includes nested structures for listings, contracts, and subsidies.
    """
    try:
        # Fetch all listings
        print("Fetching all listings...")
        listings = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_CROP_LISTINGS
        )["documents"]

        # Fetch all bids
        print("Fetching all bids...")
        bids = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_BIDS
        )["documents"]

        # Organize listings with their bids
        listings_data = {}
        for listing in listings:
            listing_id = listing["$id"]
            listings_data[listing_id] = {
                "details": listing,
                "bids": [bid for bid in bids if bid["listing_id"] == listing_id],
            }

        # Fetch all contracts
        print("Fetching all contracts...")
        contracts = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_CONTRACTS
        )["documents"]

        # Fetch all contract requests
        print("Fetching all contract requests...")
        contract_requests = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_CONTRACT_REQUESTS
        )["documents"]

        # Organize contracts with their requests
        contracts_data = {}
        for contract in contracts:
            contract_id = contract["$id"]
            contracts_data[contract_id] = {
                "details": contract,
                "requests": [
                    request
                    for request in contract_requests
                    if request["contract_id"] == contract_id
                ],
            }

        # Fetch all subsidies
        print("Fetching all subsidies...")
        subsidies = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_SUBSIDIES
        )["documents"]

        # Fetch all subsidy requests
        print("Fetching all subsidy requests...")
        subsidy_requests = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_SUBSIDY_REQUESTS
        )["documents"]

        # Organize subsidies with their requests
        subsidies_data = {}
        for subsidy in subsidies:
            subsidy_id = subsidy["$id"]
            subsidies_data[subsidy_id] = {
                "details": subsidy,
                "requests": [
                    request
                    for request in subsidy_requests
                    if request["subsidy_id"] == subsidy_id
                ],
            }

        # Combine all data
        all_data = {
            "listings": listings_data,
            "contracts": contracts_data,
            "subsidies": subsidies_data,
        }

        # Save to JSON
        with open("all_data.json", "w") as file:
            json.dump(all_data, file, indent=4)
        print("All data saved to all_data.json.")

    except Exception as e:
        print(f"Error fetching data: {e}")


# Run the function
if __name__ == "__main__":
    fetch_all_data()
