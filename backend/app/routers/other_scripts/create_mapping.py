import pandas as pd
from appwrite.id import ID
from core.config import COLLECTION_ZIPCODES, DATABASES, DATABASE_ID
from threading import Thread
from typing import List


def insert_chunk(chunk):
    """
    Inserts a chunk of rows into the COLLECTION_ZIPCODES collection.
    """
    for row in chunk:
        try:
            DATABASES.create_document(
                database_id=DATABASE_ID,
                collection_id=COLLECTION_ZIPCODES,
                document_id=ID.unique(),  # Use pincode as the document ID
                data={
                    "zipcode": str(row["pincode"]),
                    "state": row["statename"],
                    "latitude": row["latitude"],
                    "longitude": row["longitude"],
                },
            )
        except Exception as e:
            print(f"Failed to insert pincode {row['pincode']}: {e}")


def insert_zipcodes():
    """
    Reads from result.csv and inserts rows into COLLECTION_ZIPCODES using 16 threads.
    """

    # Load the CSV file
    csv_file = "routers/result.csv"  # Replace with the actual path to your CSV file
    df = pd.read_csv(csv_file)

    # Convert DataFrame rows to a list of dictionaries
    rows = df.to_dict(orient="records")

    # Define the number of threads
    num_threads = 16  # Match the number of CPU cores

    # Split rows into chunks for each thread
    chunk_size = len(rows) // num_threads
    chunks = [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]

    # Create threads for processing chunks
    threads: List[Thread] = []
    for chunk in chunks:
        thread = Thread(target=insert_chunk, args=(chunk,))
        threads.append(thread)
        thread.start()

    # Wait for all threads to complete
    for thread in threads:
        thread.join()

    return {"message": "Zipcodes inserted successfully."}
