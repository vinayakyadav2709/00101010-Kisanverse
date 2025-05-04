import os
import pandas as pd
from core.config import DATABASES, DATABASE_ID, COLLECTION_WEATHER
import json
from appwrite.query import Query


def fetch_and_store_weather_data():
    """
    Fetch all documents from COLLECTION_WEATHER and store each entry in a separate CSV file.
    """
    try:
        # Create the 'csvs' folder if it doesn't exist
        output_folder = "csvs"
        os.makedirs(output_folder, exist_ok=True)

        # Fetch all documents from the COLLECTION_WEATHER collection
        print("Fetching all weather data from the database...")
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_WEATHER,
            queries=[Query.limit(100)],
        )

        # Iterate through each document
        for document in documents["documents"]:
            latitude = document["latitude"]
            longitude = document["longitude"]
            predictions = document["predictions"]

            # Parse predictions (array of JSON strings)
            parsed_predictions = []
            for prediction in predictions:
                # Convert JSON string to dictionary
                data = json.loads(prediction)

                # Extract required fields and remove year from the month
                parsed_predictions.append(
                    {
                        "month": data["month"],  # Extract only the month
                        "temperature_2m_max": data["temperature_2m_max"],
                        "temperature_2m_min": data["temperature_2m_min"],
                        "precipitation_sum": data["precipitation_sum"],
                        "wind_speed_10m_max": data["wind_speed_10m_max"],
                        "shortwave_radiation_sum": data["shortwave_radiation_sum"],
                    }
                )

            # Sort the parsed predictions by month (ascending order)
            parsed_predictions = sorted(
                parsed_predictions, key=lambda x: int(x["month"])
            )

            # Convert the parsed predictions to a DataFrame
            df = pd.DataFrame(parsed_predictions)

            # Generate the file name based on latitude and longitude
            file_name = f"{latitude}_{longitude}.csv"
            file_path = os.path.join(output_folder, file_name)

            # Save the DataFrame to a CSV file
            df.to_csv(file_path, index=False)
            print(f"Weather data saved to {file_path}")

    except Exception as e:
        print(f"Error fetching and storing weather data: {str(e)}")


def count_unique_lat_lon():
    """
    Fetch all latitude and longitude combinations from COLLECTION_WEATHER,
    count the number of unique and duplicate combinations, and save the documents to a JSON file.
    """
    try:
        # Fetch all documents from the COLLECTION_WEATHER collection
        print("Fetching all weather data from the database...")
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_WEATHER,
        )
        print(f"Total records in the database: {documents['total']}")

        # Save the documents to a JSON file
        with open("weather_data.json", "w") as json_file:
            json.dump(documents, json_file, indent=4)
        print("Documents saved to weather_data.json")

        # Extract latitude and longitude from each document
        lat_lon_combinations = [
            (document["latitude"], document["longitude"])
            for document in documents["documents"]
        ]
        print(f"Extracted latitude and longitude combinations: {lat_lon_combinations}")

        # Convert to a DataFrame for easier processing
        df = pd.DataFrame(lat_lon_combinations, columns=["latitude", "longitude"])

        # Round latitude and longitude to avoid floating-point precision issues
        df["latitude"] = df["latitude"].round(6)
        df["longitude"] = df["longitude"].round(6)

        # Count unique combinations
        unique_count = df.drop_duplicates().shape[0]

        # Identify and count duplicate combinations
        duplicates = df.duplicated(subset=["latitude", "longitude"], keep=False)
        duplicate_count = duplicates.sum()

        # Debugging: Print duplicate rows
        if duplicate_count > 0:
            print("Duplicate rows:")
            print(df[duplicates])

        print(f"Number of unique latitude and longitude combinations: {unique_count}")
        print(
            f"Number of duplicate latitude and longitude combinations: {duplicate_count}"
        )
        return unique_count, duplicate_count

    except Exception as e:
        print(
            f"Error counting unique and duplicate latitude and longitude combinations: {str(e)}"
        )
        return 0, 0
