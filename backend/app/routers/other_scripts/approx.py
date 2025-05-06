from appwrite.query import Query
from core.config import (
    DATABASE_ID,
    COLLECTION_PRICES,
    COLLECTION_ZIPCODES,
    COLLECTION_WEATHER,
    DATABASES,
)


def round_down(value, decimals=4):
    """
    Rounds down a float to the specified number of decimal places using string manipulation.
    """
    value_str = f"{value:.{decimals + 10}f}"  # Convert to string with extra precision
    integer_part, decimal_part = value_str.split(".")
    truncated_decimal = decimal_part[
        :decimals
    ]  # Truncate to the desired number of decimals
    return float(f"{integer_part}.{truncated_decimal}")


def update_lat_lon(collection_id):
    """
    Updates the latitude and longitude fields in the specified collection
    to be accurate to 4 decimal places, rounded down.

    Args:
        collection_id (str): The ID of the collection to update.
    """
    try:
        # Fetch all documents in the collection
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=collection_id,
            queries=[Query.limit(10000000)],
        )

        for document in documents["documents"]:
            doc_id = document["$id"]
            updated_data = {}

            # Check and update latitude
            if "latitude" in document:
                updated_data["latitude"] = round_down(document["latitude"])

            # Check and update longitude
            if "longitude" in document:
                updated_data["longitude"] = round_down(document["longitude"])

            # Update the document if changes are made
            if updated_data:
                DATABASES.update_document(
                    database_id=DATABASE_ID,
                    collection_id=collection_id,
                    document_id=doc_id,
                    data=updated_data,
                )

    except Exception as e:
        print(f"Error updating collection {collection_id}: {str(e)}")


def main():
    """
    Main function to update latitude and longitude in all relevant collections.
    """
    collections = [COLLECTION_ZIPCODES]

    for collection_id in collections:
        print(f"Updating collection: {collection_id}")
        update_lat_lon(collection_id)
