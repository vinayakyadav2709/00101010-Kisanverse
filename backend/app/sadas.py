import json
from core.config import (
    DATABASE_ID,
    COLLECTION_BIDS,
    COLLECTION_SUBSIDIES,
    DATABASES,
    COLLECTION_SUBSIDY_REQUESTS,
    COLLECTION_CONTRACTS,
    COLLECTION_USERS,
    COLLECTION_CROP_LISTINGS,
)
from datetime import datetime
from appwrite.query import Query

# import routers.ai as ai


def fetch_accepted_bids():
    """
    Fetch all bids with status 'accepted' and store them in bids.json.
    """
    try:
        print("Fetching all accepted bids...")
        bids = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_BIDS,
            queries=[Query.equal("status", "accepted")],
        )["documents"]

        # Save the results to bids.json
        with open("bids.json", "w") as file:
            json.dump(bids, file, indent=4)
        print("Accepted bids saved to bids.json.")
    except Exception as e:
        print(f"Error fetching accepted bids: {e}")


def fetch_recent_subsidies():
    """
    Fetch the most recent 3 subsidies and store them in subsidy.json.
    """
    try:
        print("Fetching the most recent 3 subsidies...")
        subsidies = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_SUBSIDIES,
            queries=[Query.order_desc("$createdAt"), Query.limit(3)],
        )["documents"]

        # Save the results to subsidy.json
        with open("subsidy.json", "w") as file:
            json.dump(subsidies, file, indent=4)
        print("Most recent subsidies saved to subsidy.json.")
    except Exception as e:
        print(f"Error fetching recent subsidies: {e}")


from typing import Dict, Any


def fetch_weather_and_prices():
    """
    Fetch weather and price data for the given latitude, longitude, crop, and date range.
    """
    try:
        # Define parameters
        latitude = 21.5929
        longitude = 81.3761
        crop = "WHEAT"
        start_date = "2025-05-04"
        end_date = "2025-10-04"

        # Fetch weather data
        print("Fetching weather data...")
        weather_data = ai.get_weather(latitude, longitude, start_date, end_date)
        print("Weather data fetched successfully.")

        # Fetch price data
        print("Fetching price data...")
        price_data = ai.fetch_prices(latitude, longitude, crop, start_date, end_date)
        print("Price data fetched successfully.")

        # Combine results
        result = {
            "weather": weather_data,
            "prices": price_data,
        }

        # Save the result to a JSON file
        with open("weather_and_prices.json", "w") as file:
            json.dump(result, file, indent=4)
        print("Weather and price data saved to weather_and_prices.json.")

        return result

    except Exception as e:
        print(f"Error fetching weather and price data: {e}")
        raise


def fetch_subsidy_requests():
    """
    Fetch all subsidy requests and extract relevant fields from the corresponding subsidies.
    Store the results in subsidy_requests.json.
    """
    try:
        print("Fetching all subsidy requests...")
        # Fetch all subsidy requests
        subsidy_requests = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_SUBSIDY_REQUESTS,  # Replace with the actual collection ID for subsidy requests
        )["documents"]

        if not subsidy_requests:
            print("No subsidy requests found.")
            return

        # Prepare the result
        result = []
        for request in subsidy_requests:
            subsidy_id = request.get("subsidy_id")
            applied_on = request.get("$createdAt")
            status = request.get("status")
            # Fetch the corresponding subsidy
            subsidy = DATABASES.get_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_SUBSIDIES,
                document_id=subsidy_id,
            )
            name = subsidy.get("program")
            # Extract required fields
            result.append(
                {
                    "program": name,
                    "benefits": subsidy.get("benefits"),
                    "type": subsidy.get("type"),
                    "applied_on": applied_on,
                    "status": status,
                }
            )

        # Save the result to a JSON file
        with open("subsidy_requests.json", "w") as file:
            json.dump(result, file, indent=4)
        print("Subsidy requests saved to subsidy_requests.json.")

    except Exception as e:
        print(f"Error fetching subsidy requests: {e}")


def fetch_contracts():
    """
    Fetch all contracts and create a JSON file with the specified fields.
    """
    try:
        print("Fetching all contracts...")
        # Fetch all contracts
        contracts = DATABASES.list_documents(
            database_id=DATABASE_ID, collection_id=COLLECTION_CONTRACTS
        )["documents"]

        if not contracts:
            print("No contracts found.")
            return

        # Prepare the result
        result = []
        for contract in contracts:
            buyer_id = contract.get("buyer_id")
            # Fetch the buyer's name using buyer_id
            buyer = DATABASES.get_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_USERS,
                document_id=buyer_id,
            )
            buyer_name = buyer.get("name", "Unknown Buyer")

            # Parse dynamic_fields as JSON
            dynamic_fields = json.loads(contract.get("dynamic_fields", "{}"))

            # Extract delivery_date and format it to only show the date part
            delivery_date = contract.get("delivery_date", "")
            if delivery_date:
                delivery_date = datetime.fromisoformat(delivery_date).date().isoformat()

            # Add the required fields to the result
            result.append(
                {
                    "name": buyer_name,
                    "status": contract.get("status"),
                    "dynamic_fields": dynamic_fields,
                    "locations": contract.get("locations"),
                    "crop_type": contract.get("crop_type"),
                    "quantity": contract.get("quantity"),
                    "advance_payment": contract.get("advance_payment"),
                    "delivery_date": delivery_date,
                    "payment_terms": contract.get("payment_terms"),
                }
            )

        # Save the result to a JSON file
        with open("contracts.json", "w") as file:
            json.dump(result, file, indent=4)
        print("Contracts saved to contracts.json.")

    except Exception as e:
        print(f"Error fetching contracts: {e}")


# Example usage
if __name__ == "__main__":
    fetch_contracts()
