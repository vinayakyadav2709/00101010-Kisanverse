import pandas as pd
import time
from datetime import datetime, timedelta
import json
from models import weather
from core.config import DATABASES, DATABASE_ID, COLLECTION_WEATHER
from appwrite.id import ID
from appwrite.query import Query


def convert_to_serializable(obj):
    """
    Helper function to convert non-serializable types to serializable ones.
    """
    if isinstance(obj, (datetime, timedelta)):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} is not JSON serializable")


def fetch_weather_data():
    try:
        # Read latitude and longitude from a CSV file
        csv_file = "routers/unique_lat_lon.csv"  # Replace with your CSV file path
        print(f"Reading coordinates from {csv_file}...")
        df = pd.read_csv(csv_file)

        # Convert DataFrame rows to a list of dictionaries
        coordinates = df.to_dict(orient="records")
        print(f"Loaded {len(coordinates)} coordinates from the CSV file.")

        # Start date as tomorrow
        start_date = datetime.now().date() + timedelta(days=1)
        print(f"Start date for fetching weather data: {start_date}")

        # Loop through each coordinate
        for coord in coordinates:
            lat = coord["latitude"]
            lon = coord["longitude"]
            print(f"Processing coordinate: Latitude={lat}, Longitude={lon}")

            # Check if the latitude and longitude already exist in the database
            try:
                print(
                    f"Checking if data for Latitude={lat}, Longitude={lon} exists in the database..."
                )
                existing_data = DATABASES.list_documents(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_WEATHER,
                    queries=[
                        Query.equal("latitude", lat),
                        Query.equal("longitude", lon),
                    ],
                )
                if existing_data["total"] > 0:
                    print(
                        f"Data for Latitude={lat}, Longitude={lon} already exists. Skipping..."
                    )
                    continue  # Skip this coordinate if data already exists
            except Exception as db_error:
                print(
                    f"Error querying database for Latitude={lat}, Longitude={lon}: {db_error}"
                )
                continue

            # Prepare to fetch data for the next 9 months
            predictions = []
            for month_offset in range(9):
                # Calculate the date for each month
                current_date = start_date + timedelta(days=month_offset * 30)
                current_month = current_date.strftime("%Y-%m")  # Extract year and month
                print(f"Fetching weather data for month: {current_month}")

                # Fetch weather data with exponential backoff
                wait_time = 60  # Start with 60 seconds
                for retry in range(5):  # Retry up to 5 times
                    try:
                        print(
                            f"Attempting to fetch weather data (Retry {retry + 1})..."
                        )
                        forecast = weather.get_enhanced_forecast_on_end_date(
                            lat,
                            lon,
                            start_date.strftime("%Y-%m-%d"),
                            current_date.strftime("%Y-%m-%d"),
                        )
                        # Validate forecast is a dictionary
                        if not isinstance(forecast, dict):
                            raise ValueError(f"Invalid forecast data: {forecast}")

                        # Round values to 2 decimal places
                        for key in forecast:
                            if isinstance(forecast[key], float):
                                forecast[key] = round(forecast[key], 2)

                        # Add the month to the forecast data
                        forecast["month"] = current_month

                        # Serialize forecast and append to predictions
                        serialized_forecast = json.dumps(
                            forecast, default=convert_to_serializable
                        )
                        predictions.append(serialized_forecast)
                        print(f"Successfully fetched weather data for {current_month}.")
                        break  # Exit the retry loop on success
                    except Exception as e:
                        if "429" in str(e):  # Handle API rate limit error
                            retry_after = 60  # Default to 60 seconds
                            if (
                                hasattr(e, "response")
                                and "Retry-After" in e.response.headers
                            ):
                                retry_after = int(e.response.headers["Retry-After"])
                            print(
                                f"Rate limit exceeded. Waiting for {retry_after} seconds..."
                            )
                            time.sleep(retry_after + 5)  # Add a buffer
                            wait_time *= 2  # Exponential backoff
                        else:
                            print(
                                f"Error fetching forecast for Latitude={lat}, Longitude={lon}: {e}"
                            )
                            raise e  # Raise other exceptions

            # Store predictions in the COLLECTION_WEATHER collection
            try:
                print(
                    f"Storing weather data for Latitude={lat}, Longitude={lon} in the database..."
                )
                DATABASES.create_document(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_WEATHER,
                    document_id=ID.unique(),
                    data={
                        "latitude": lat,
                        "longitude": lon,
                        "predictions": predictions,  # Store predictions as an array of JSON strings
                    },
                )
                print(
                    f"Weather data for Latitude={lat}, Longitude={lon} stored successfully."
                )
            except Exception as db_error:
                print(
                    f"Failed to store weather data for Latitude={lat}, Longitude={lon}: {db_error}"
                )

    except Exception as e:
        print(f"Error fetching weather data: {str(e)}")


def adjust_weather_data():
    try:
        print("Fetching all weather documents with pagination...")

        # Fetch all documents using pagination

        all_documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_WEATHER,
            queries=[Query.limit(100)],  # Limit the fetch to 100 documents
        )["documents"]

        print(f"Total documents fetched: {len(all_documents)}")

        # Now adjust each document
        updated_count = 0

        for document in all_documents:
            document_id = document["$id"]
            predictions = document["predictions"]  # Array of JSON strings

            updated_predictions = []
            found_first_10 = False
            modified = False  # Track whether anything was changed

            for prediction in predictions:
                data = json.loads(prediction)

                # Remove the 'date' field if it exists
                if "date" in data:
                    del data["date"]
                    modified = True

                # Normalize 'month' to two-digit string (remove year if present)
                if "month" in data:
                    original_month = data["month"]
                    if "-" in original_month:
                        current_month = int(original_month.split("-")[1])
                        modified = True
                    else:
                        current_month = int(original_month)

                    if current_month == 10 and not found_first_10:
                        found_first_10 = True
                    elif found_first_10:
                        current_month += 1
                        modified = True

                    data["month"] = f"{current_month:02d}"

                updated_predictions.append(json.dumps(data))

            # Update only if there was a change
            if modified:
                DATABASES.update_document(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_WEATHER,
                    document_id=document_id,
                    data={"predictions": updated_predictions},
                )
                updated_count += 1
                print(f"Updated document: {document_id}")

        print(f"Total documents updated: {updated_count}")

    except Exception as e:
        print(f"Error adjusting weather data: {str(e)}")


def printt():
    print(
        DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_WEATHER,
        )
    )
