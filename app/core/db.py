# note: this is old verison. needs a lot of changes.

from appwrite.client import Client
from appwrite.services.users import Users
from appwrite.id import ID

from appwrite.services.databases import Databases
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()
client = Client()
client = (
    client.set_endpoint("http://localhost/v1")
    .set_project(os.environ["APPWRITE_PROJECT_ID"])
    .set_key(os.getenv("APPWRITE_API_KEY"))
    .set_self_signed(True)
)
databases = Databases(client)
database_id = "agri_marketplace"

users = Users(client)


def create_db():
    # Check if the database already exists
    try:
        databases.get(database_id)
        print(f"Database '{database_id}' already exists.")
    except:
        databases.create(database_id=database_id, name="Agri Marketplace")
        print(f"Database '{database_id}' created.")


create_db()
collections = {
    "users": {
        "name": "Users",
        "attributes": [
            ("name", "string", 50, True),
            ("phone", "string", 15, True),
            ("role", "enum", ["farmer", "buyer", "admin", "provider"], True),
            ("address", "string", 200, False),
        ],
    },
    "crop_listings": {
        "name": "Crop Listings",
        "attributes": [
            ("farmer_id", "string", 36, True),
            ("farmer_address", "string", 200, False),
            ("crop_type", "string", 50, True),
            ("price_per_kg", "float", True),
            ("total_quantity", "float", True),
            ("available_quantity", "float", True),
            (
                "status",
                "enum",
                ["fulfilled", "cancelled", "listed", "removed"],
                False,
                "listed",
            ),
        ],
    },
    "bids": {
        "name": "Bids",
        "attributes": [
            ("listing_id", "string", 36, True),
            ("buyer_id", "string", 36, True),
            ("quantity", "float", True),
            ("price_per_kg", "float", True),
            (
                "status",
                "enum",
                [
                    "pending",
                    "accepted",
                    "rejected",
                    "withdrawn",
                    "fulfilled",
                    "removed",
                ],
                False,
                "pending",
            ),
            ("timestamp", "datetime", True),
        ],
    },
    "contracts": {
        "name": "Contracts",
        "attributes": [
            ("buyer_id", "string", 36, True),
            ("template_type", "enum", ["seed", "machine", "fert", "general"], True),
            (
                "status",
                "enum",
                ["listed", "accepted", "cancelled", "removed", "fulfilled"],
                False,
                "listed",
            ),
            ("dynamic_fields", "string", 5000, False),
            ("location", "string", 100, True, None, True),
        ],
    },
    "contract_requests": {
        "name": "Contract Requests",
        "attributes": [
            ("contract_id", "string", 36, True),
            ("farmer_id", "string", 36, True),
            (
                "status",
                "enum",
                [
                    "pending",
                    "accepted",
                    "rejected",
                    "withdrawn",
                    "fulfilled",
                    "removed",
                ],
                False,
                "pending",
            ),
        ],
    },
    "subsidies": {
        "name": "Subsidies",
        "attributes": [
            ("submitted_by", "string", 36, True),
            (
                "type",
                "enum",
                ["crop", "seed", "fertilizer", "machine", "general"],
                True,
            ),
            ("location", "string", 100, True, None, True),  # array=True
            ("max_recipients", "integer", True),
            (
                "status",
                "enum",
                ["pending", "approved", "rejected", "removed"],
                False,
                "pending",
            ),
            ("dynamic_fields", "string", 5000, False),
        ],
    },
    "subsidy_requests": {
        "name": "Subsidy Requests",
        "attributes": [
            ("subsidy_id", "string", 36, True),
            ("farmer_id", "string", 36, True),
            (
                "status",
                "enum",
                ["requested", "accepted", "rejected", "withdrawn", "removed"],
                False,
                "requested",
            ),
        ],
    },
}


def create_collection_and_attributes():
    for collection_id, meta in collections.items():
        try:
            databases.create_collection(
                database_id=database_id,
                collection_id=collection_id,
                name=meta["name"],
                permissions=[],
                document_security=False,
            )
            print(f"✅ Created collection: {collection_id}")
        except Exception as e:
            print(f"⚠️ Collection {collection_id} might already exist: {e}")

        for attr in meta["attributes"]:
            try:
                key, type_, *args = attr
                array = False
                default = None

                # Handle array-type workaround in definition (e.g. ("location", "array", "string", True))
                if type_ == "array":
                    base_type, required = args
                    type_ = base_type
                    array = True
                elif len(args) > 0 and args[-1] == "array":
                    array = True
                    args = args[:-1]

                if type_ == "string":
                    size = args[0]
                    required = args[1]
                    databases.create_string_attribute(
                        database_id,
                        collection_id,
                        key=key,
                        size=size,
                        required=required,
                        array=array,
                    )
                elif type_ == "float":
                    required = args[0]
                    databases.create_float_attribute(
                        database_id,
                        collection_id,
                        key=key,
                        required=required,
                        array=array,
                    )
                elif type_ == "integer":
                    required = args[0]
                    databases.create_integer_attribute(
                        database_id,
                        collection_id,
                        key=key,
                        required=required,
                        array=array,
                    )
                elif type_ == "datetime":
                    required = args[0]
                    databases.create_datetime_attribute(
                        database_id,
                        collection_id,
                        key=key,
                        required=required,
                        array=array,
                    )
                elif type_ == "enum":
                    elements = args[0]
                    required = args[1]
                    default = args[2] if len(args) > 2 else None
                    databases.create_enum_attribute(
                        database_id,
                        collection_id,
                        key=key,
                        elements=elements,
                        required=required,
                        default=default,
                    )

                print(f"  ➕ Added attribute: {key} (array={array})")
            except Exception as e:
                print(f"⚠️ Attribute {key} might already exist or error: {e}")


def delete_all_collections():
    try:
        result = databases.list_collections(database_id)
        for collection in result["collections"]:
            collection_id = collection["$id"]
            databases.delete_collection(database_id, collection_id)
            print(f"🗑️ Deleted collection: {collection_id}")
    except Exception as e:
        print(f"❌ Error deleting collections: {e}")


# delete_all_collections()
create_collection_and_attributes()
# not done: address required. index for phone number of type unique
# added at,updated at for listings
# removed farmer address from crop listings
# change phone to email
# add fulfilled,withdrawn in subsidy
# cahnge location to locations in subsidies
# change location to locations in contracts
