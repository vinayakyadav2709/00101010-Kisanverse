from asyncio import base_events
import os
import json
from datetime import datetime, timezone
from core.config import DATABASES, DATABASE_ID, COLLECTION_PRICES
from appwrite.id import ID


def insert_prices_from_files():
    """
    Iterate through folders and files in the base folder, extract crop price data,
    and insert it into the COLLECTION_PRICES table.

    Args:
        base_folder (str): The base folder containing the lat_long folders.
    """
    base_folder = "models/prices"
    try:
        # Iterate through each folder in the base folder
        for folder_name in os.listdir(base_folder):
            folder_path = os.path.join(base_folder, folder_name)

            # Skip if it's not a directory
            if not os.path.isdir(folder_path):
                continue

            # Extract latitude and longitude from the folder name
            try:
                latitude, longitude = map(float, folder_name.split("_"))
            except ValueError:
                print(f"Skipping folder {folder_name}: Invalid lat_long format.")
                continue

            # Iterate through each file in the folder
            for file_name in os.listdir(folder_path):
                if file_name == "crop_predictions_simple_processed.json":
                    file_path = os.path.join(folder_path, file_name)

                    # Read the JSON file
                    with open(file_path, "r") as f:
                        crop_data = json.load(f)

                    # Iterate through each crop in the JSON data
                    for crop, details in crop_data.items():
                        dates = details.get("dates", [])
                        prices = details.get("prices_kg", [])

                        # Ensure dates and prices have the same length
                        if len(dates) != len(prices):
                            min_length = min(len(dates), len(prices))
                            dates = dates[:min_length]
                            prices = prices[:min_length]

                        # Insert each date and price into the COLLECTION_PRICES table

                        document_data = {
                            "latitude": latitude,
                            "longitude": longitude,
                            "crop": crop,
                            "dates": dates,  # Use current UTC time
                            "prices": prices,
                        }

                        # Insert the document into the database
                        try:
                            DATABASES.create_document(
                                database_id=DATABASE_ID,
                                collection_id=COLLECTION_PRICES,
                                document_id=ID.unique(),
                                data=document_data,
                            )
                            print(
                                f"Inserted data for crop {crop} at {latitude}, {longitude}: {document_data}"
                            )
                        except Exception as e:
                            print(
                                f"Error inserting data for crop {crop} at {latitude}, {longitude}: {str(e)}"
                            )

    except Exception as e:
        print(f"Error processing files: {str(e)}")


from appwrite.query import Query


def update_crop_names_to_uppercase():
    """
    Fetch all documents from the COLLECTION_PRICES collection and update the crop name to uppercase.
    """
    try:
        # Fetch documents in batches of 5000
        # Fetch a batch of documents
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_PRICES,
            queries=[
                Query.limit(50000),
            ],
        )

        # If no documents are returned, break the loop
        if not documents["documents"]:
            print("No documents found.")
            return

        # Iterate through each document and update the crop name
        for document in documents["documents"]:
            document_id = document["$id"]
            crop_name = document.get("crop", "")
            updated_crop_name = crop_name.upper()

            # Update the document with the uppercase crop name
            try:
                DATABASES.update_document(
                    database_id=DATABASE_ID,
                    collection_id=COLLECTION_PRICES,
                    document_id=document_id,
                    data={"crop": updated_crop_name},
                )
                print(f"Updated crop name to uppercase for document ID: {document_id}")
            except Exception as e:
                print(f"Error updating document ID {document_id}: {str(e)}")

    except Exception as e:
        print(f"Error updating crop names: {str(e)}")
