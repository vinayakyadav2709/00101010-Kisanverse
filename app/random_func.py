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
from routers import ai


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

import json
from datetime import datetime, timedelta
from models import weather


def get_rainfall_description(rainfall):
    """
    Categorize rainfall into low, medium, or high based on its value.
    """
    if rainfall == 0:
        return "none"
    elif rainfall < 2.5:
        return "low"
    elif rainfall < 7.6:
        return "medium"
    else:
        return "high"


def fetch_weather_data_for_coordinate(latitude, longitude):
    try:
        # Start date as today
        start_date = datetime.now().date()
        print(f"Fetching weather data for Latitude={latitude}, Longitude={longitude}")
        print(f"Start date: {start_date}")

        # Prepare to fetch data for the next 16 days
        weather_summary = []
        for day_offset in range(16):
            # Calculate the date for each day
            current_date = start_date + timedelta(days=day_offset)
            print(f"Fetching weather data for date: {current_date}")

            # Fetch weather data
            try:
                forecast = weather.get_enhanced_forecast_on_end_date(
                    latitude,
                    longitude,
                    current_date.strftime("%Y-%m-%d"),
                    current_date.strftime("%Y-%m-%d"),
                )
                # Validate forecast is a dictionary
                if not isinstance(forecast, dict):
                    raise ValueError(f"Invalid forecast data: {forecast}")

                # Extract required fields
                date = current_date.strftime("%Y-%m-%d")
                rainfall = round(forecast.get("precipitation_sum", 0), 2)
                rainfall_description = get_rainfall_description(rainfall)
                wind_speed = round(forecast.get("wind_speed_10m_max", 0), 2)
                temperature = round(
                    (
                        forecast.get("temperature_2m_max", 0)
                        + forecast.get("temperature_2m_min", 0)
                    )
                    / 2,
                    2,
                )

                # Append the summarized data
                weather_summary.append(
                    {
                        "date": date,
                        "rainfall": rainfall,
                        "rainfall_description": rainfall_description,
                        "wind_speed": wind_speed,
                        "temperature": temperature,
                    }
                )
                print(f"Successfully fetched weather data for {current_date}.")
            except Exception as e:
                print(
                    f"Error fetching forecast for Latitude={latitude}, Longitude={longitude} on {current_date}: {e}"
                )
                continue

        # Save the weather summary to a JSON file
        output_file = f"weather.json"
        with open(output_file, "w") as json_file:
            json.dump(weather_summary, json_file, indent=4)
        print(f"Weather summary saved to {output_file}")

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")


# Run the function
if __name__ == "__main__":
    # Replace with the desired latitude and longitude
    latitude = 28.7041  # Example: Latitude for Delhi
    longitude = 77.1025  # Example: Longitude for Delhi
    fetch_weather_data_for_coordinate(latitude, longitude)
