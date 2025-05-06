import json
from core.config import (
    DATABASES,
    DATABASE_ID,
    STORAGE,
)
from appwrite.id import ID
import os


def delete_all_collections():
    """
    Deletes all collections in the database.
    """
    print("Deleting all collections...")
    try:
        collections = DATABASES.list_collections(database_id=DATABASE_ID)["collections"]
        for collection in collections:
            collection_id = collection["$id"]
            DATABASES.delete_collection(
                database_id=DATABASE_ID, collection_id=collection_id
            )
            print(f"Deleted collection: {collection_id}")
    except Exception as e:
        print(f"Error deleting collections: {e}")


def delete_all_buckets():
    """
    Deletes all buckets in storage.
    """
    print("Deleting all buckets...")
    try:
        buckets = STORAGE.list_buckets()["buckets"]
        for bucket in buckets:
            bucket_id = bucket["$id"]
            STORAGE.delete_bucket(bucket_id=bucket_id)
            print(f"Deleted bucket: {bucket_id}")
    except Exception as e:
        print(f"Error deleting buckets: {e}")


def import_collections():
    """
    Import all documents into the specified collections from the exported JSON file.
    """
    print("Importing collections...")
    with open("migration/collections_data.json", "r") as file:
        collections_data = json.load(file)
    for collection, documents in collections_data.items():
        print(f"Creating collection: {collection}")
        try:
            DATABASES.create_collection(
                database_id=DATABASE_ID,
                collection_id=collection,
                name=collection,
                permissions=[],  # Add appropriate permissions if needed
                document_security=True,
            )
            for document in documents:
                DATABASES.create_document(
                    database_id=DATABASE_ID,
                    collection_id=collection,
                    document_id=document["$id"],  # Use the same document ID
                    data=document,
                )
            print(f"Imported {len(documents)} documents into {collection}")
        except Exception as e:
            print(f"Error importing collection {collection}: {e}")


def import_buckets():
    """
    Import all files into the specified buckets from the exported files.
    """
    print("Importing buckets...")
    with open("migration/buckets_data.json", "r") as file:
        buckets_data = json.load(file)
    for bucket, files in buckets_data.items():
        print(f"Creating bucket: {bucket}")
        try:
            STORAGE.create_bucket(
                bucket_id=bucket,
                name=bucket,
                permissions=[],  # Add appropriate permissions if needed
                file_security=True,
            )
            for file_name in files:
                file_path = f"migration/{bucket}_{file_name}"
                if not os.path.exists(file_path):
                    print(
                        f"Warning: File {file_name} for bucket {bucket} is missing. Skipping..."
                    )
                    continue
                with open(file_path, "rb") as f:
                    STORAGE.create_file(bucket_id=bucket, file_id=ID.unique(), file=f)
            print(f"Imported {len(files)} files into {bucket}")
        except Exception as e:
            print(f"Error importing bucket {bucket}: {e}")


if __name__ == "__main__":
    print("Starting import process...")
    delete_all_collections()
    delete_all_buckets()
    import_collections()
    import_buckets()
    print("Import process completed successfully!")
