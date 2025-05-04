import json
from core.config import (
    DATABASE_ID,
    COLLECTION_BIDS,
    COLLECTION_SUBSIDIES,
    DATABASES,
)
from appwrite.query import Query


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


if __name__ == "__main__":
    fetch_accepted_bids()
    fetch_recent_subsidies()
