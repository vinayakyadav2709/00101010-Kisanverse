import json
from core.config import DATABASES, DATABASE_ID, STORAGE
from appwrite.query import Query
import urllib3

# Suppress InsecureRequestWarning
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


def list_all_collections():
    """
    List all collections in the database.
    """
    print("Fetching all collections...")
    collections = DATABASES.list_collections(database_id=DATABASE_ID)["collections"]
    collection_ids = [collection["$id"] for collection in collections]
    print(f"Found {len(collection_ids)} collections: {collection_ids}")
    return collection_ids


def list_all_buckets():
    """
    List all storage buckets.
    """
    print("Fetching all buckets...")
    buckets = STORAGE.list_buckets()["buckets"]
    bucket_ids = [bucket["$id"] for bucket in buckets]
    print(f"Found {len(bucket_ids)} buckets: {bucket_ids}")
    return bucket_ids


def export_collections():
    """
    Export all documents from all collections and save them to a JSON file.
    """
    print("Exporting collections...")
    exported_data = {}
    collections = list_all_collections()
    for collection in collections:
        print(f"Exporting collection: {collection}")
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=collection,
            queries=[Query.limit(10000000)],
        )["documents"]
        exported_data[collection] = documents
        print(f"Exported {len(documents)} documents from {collection}")
    with open("migration/collections_data.json", "w") as file:
        json.dump(exported_data, file, indent=4)
    print("Exported all collections to collections_data.json")


def export_buckets():
    """
    Export all files from all buckets and save them locally.
    """
    print("Exporting buckets...")
    exported_buckets = {}
    buckets = list_all_buckets()
    for bucket in buckets:
        print(f"Exporting bucket: {bucket}")
        files = STORAGE.list_files(bucket_id=bucket)["files"]
        bucket_files = []
        for file in files:
            file_data = STORAGE.get_file_download(bucket_id=bucket, file_id=file["$id"])
            with open(f"migration/{bucket}_{file['name']}", "wb") as f:
                f.write(file_data)
            bucket_files.append(file["name"])
        exported_buckets[bucket] = bucket_files
        print(f"Exported {len(bucket_files)} files from {bucket}")
    with open("migration/buckets_data.json", "w") as file:
        json.dump(exported_buckets, file, indent=4)
    print("Exported all buckets to buckets_data.json")


if __name__ == "__main__":
    print("Starting export process...")
    export_collections()
    export_buckets()
    print("Export process completed successfully!")
