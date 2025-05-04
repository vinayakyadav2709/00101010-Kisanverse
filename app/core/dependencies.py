from fastapi import HTTPException
from core.config import DATABASES, DATABASE_ID, COLLECTION_USERS, COLLECTION_ZIPCODES
from appwrite.query import Query


def get_user_by_email(email: str | None) -> dict | None:
    if not email:
        return None  # Return None if email is not provided
    users = DATABASES.list_documents(
        DATABASE_ID, COLLECTION_USERS, queries=[Query.equal("email", [email])]
    )
    if not users["documents"]:
        return None  # Return None if no user is found
    return users["documents"][0]  # assuming email is unique


def get_user_by_email_or_raise(email: str | None) -> dict:
    user = get_user_by_email(email)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user


def get_document_or_raise(
    collection_id: str, document_id: str, detail: str = "Document not found"
) -> dict:
    try:
        document = DATABASES.get_document(DATABASE_ID, collection_id, document_id)
    except Exception as e:
        raise HTTPException(status_code=404, detail=detail)
    return document


# Function to fetch data from the collection
def get_coord(zipcode: str):
    try:
        val = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_ZIPCODES,
            queries=[Query.equal("zipcode", [zipcode])],
        )

        if val["documents"]:
            val = val["documents"][0]
            return {
                "latitude": val["latitude"],
                "longitude": val["longitude"],
            }
        else:
            raise HTTPException(status_code=404, detail="Given zipcode not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching zipcodes: {str(e)}"
        )


# Function to fetch data from the collection
def get_state(zipcode: str):
    try:
        val = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_ZIPCODES,
            queries=[Query.equal("zipcode", [zipcode])],
        )

        if val["documents"]:
            return val["documents"][0]["state"].upper()
        else:
            raise HTTPException(status_code=404, detail="Given zipcode not found")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching zipcodes: {str(e)}"
        )
