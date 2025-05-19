from fastapi import HTTPException
from core.config import DATABASES, DATABASE_ID, COLLECTION_USERS, COLLECTION_ZIPCODES
from appwrite.query import Query

from models.llm.recommendation_api import get_translations, translate_string


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


import json


def string_to_json(obj):
    """
    Recursively parses any string value in the input dict/list that is itself valid JSON,
    and replaces it with the parsed JSON object.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for k, v in obj.items():
            if isinstance(v, str):
                try:
                    parsed = json.loads(v)
                    # If parsing succeeds and result is dict or list, replace
                    if isinstance(parsed, (dict, list)):
                        new_obj[k] = string_to_json(parsed)
                    else:
                        new_obj[k] = v
                except (json.JSONDecodeError, TypeError):
                    new_obj[k] = v
            else:
                new_obj[k] = string_to_json(v)
        return new_obj
    elif isinstance(obj, list):
        return [string_to_json(item) for item in obj]
    else:
        return obj


def translate_json(json_data, language="english"):
    """
    Translate the JSON data to a different format.
    This is a placeholder function and should be implemented as needed.
    """
    try:
        language = language.lower() if language else "english"
        result = get_translations(string_to_json(json_data), language)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error translating JSON: {str(e)}")


# You can place this anywhere in your codebase, e.g., utils.py
