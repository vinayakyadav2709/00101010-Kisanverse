import re
from fastapi import APIRouter, HTTPException
from matplotlib.style import available
from pydantic import BaseModel, Field
from typing import Dict, Any
import requests
from fastapi import APIRouter, HTTPException, UploadFile, File
from appwrite.id import ID
from models.llm import recommendation_api
from appwrite.input_file import InputFile
from core.config import (
    COLLECTION_CONTRACTS,
    DATABASE_ID,
    DATABASES,
    COLLECTION_CROP_LISTINGS,
    COLLECTION_BIDS,
    COLLECTION_ZIPCODES,
    COLLECTION_SUBSIDY_REQUESTS,
    STORAGE,
    COLLECTION_DISEASE,
    BUCKET_DISEASE,
    BUCKET_SOIL,
    COLLECTION_SOIL,
    COLLECTION_WEATHER,
    COLLECTION_WEATHER_HISTORY,
    COLLECTION_USERS,
    COLLECTION_SUBSIDIES,
    COLLECTION_PRICES,
    COLLECTION_PRICES_HISTORY,
    COLLECTION_CROPS,
    BUCKET_CROP,
    PROJECT_ID,
    ENDPOINT,
)
from routers import contracts, subsidies
from core.dependencies import get_user_by_email_or_raise, get_coord
from typing import List, Optional
from pydantic import BaseModel
from appwrite.query import Query
from datetime import datetime, timezone, timedelta
from models import disease, soil

# import price
import json
from torchvision import transforms
from PIL import Image
import torch

import os

# Define the router
ai_router = APIRouter(prefix="/predictions", tags=["Predictions"])
import numpy as np  # Add this import if not already present


def get_url(bucket_id: str, file_id: str) -> str:
    return f"{ENDPOINT}/storage/buckets/{bucket_id}/files/{file_id}/view?project={PROJECT_ID}"


# Helper function to convert non-serializable types
def convert_to_serializable(obj):
    if isinstance(obj, np.float32):  # Check for numpy float32
        return float(obj)  # Convert to standard Python float
    if isinstance(obj, np.ndarray):  # Check for numpy arrays
        return obj.tolist()  # Convert arrays to lists
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@ai_router.post("/soil_type")
def soil_classification(email: str, file: UploadFile = File(...), store: bool = True):
    try:
        # Step 1: Get the user ID from the email
        user_id = get_user_by_email_or_raise(email)["$id"]

        # Step 2: Save the uploaded file locally
        local_file_path = f"/tmp/{file.filename}"
        with open(local_file_path, "wb") as f:
            f.write(file.file.read())  # Use file.file.read() for synchronous reading

        # Step 3: Perform soil classification using the refactored service
        prediction_result = soil.soil_prediction_service.predict(local_file_path)

        # Extract prediction details
        soil_type = prediction_result["predicted_class"]
        confidence = prediction_result["confidence"] * 100  # Convert to percentage
        if not store:
            # If store is False, return the prediction result without storing
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return {
                "soil_type": soil_type,
                "confidence": confidence,
            }
        # Step 4: Upload the file to Appwrite Storage
        response = STORAGE.create_file(
            bucket_id=BUCKET_SOIL,  # Replace with your Appwrite bucket ID
            file_id=ID.unique(),  # Generate a unique file ID
            file=InputFile.from_path(
                local_file_path
            ),  # Open the file in read-binary mode
        )
        file_id = response["$id"]

        # Step 5: Store metadata in the Appwrite Collection
        document = DATABASES.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_SOIL,  # Replace with your collection ID
            document_id=ID.unique(),  # Generate a unique document ID
            data={
                "user_id": user_id,
                "file_id": file_id,
                "soil_type": soil_type,
                "confidence": confidence,
                "uploaded_at": datetime.now(
                    timezone.utc
                ).isoformat(),  # Store current timestamp
            },
        )

        # Step 6: Delete the local file after processing
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        # Step 7: Return the response
        return document

    except Exception as e:
        # Ensure the local file is deleted even if an error occurs
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error during soil classification: {str(e)}"
        )


@ai_router.get("/soil_type/history")
def soil_classification_history(email: Optional[str] = None):
    """
    API endpoint to retrieve the soil classification history for a user or all users if email is not provided.

    Args:
        email (str, optional): The email of the user. If not provided, returns all entries.

    Returns:
        dict: A list of soil classification history records.
    """
    try:
        # Step 1: Query the Appwrite database
        queries = []
        if email:
            user_id = get_user_by_email_or_raise(email)["$id"]
            queries.append(Query.equal("user_id", user_id))
        queries.append(Query.limit(10000))
        documents = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SOIL,  # Replace with your collection ID
            queries=queries,  # Apply user filter if email is provided
        )

        # Step 2: Extract and return the history
        history = []
        for doc in documents["documents"]:
            file_id = doc["file_id"]
            file_url = get_url(BUCKET_SOIL, file_id)  # Generate file view URL
            history.append(
                {
                    "file_id": file_id,
                    "file_url": file_url,
                    "soil_type": doc["soil_type"],
                    "confidence": doc["confidence"],
                    "uploaded_at": doc["uploaded_at"],
                }
            )

        # Sort history by uploaded_at in descending order
        history.sort(key=lambda x: x["uploaded_at"])
        return {"history": history}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


@ai_router.post("/disease")
def disease_prediction(email: str, file: UploadFile = File(...), store: bool = True):
    try:
        user_id = get_user_by_email_or_raise(email)["$id"]

        # Step 1: Read the file contents
        local_file_path = f"/tmp/{file.filename}"
        with open(local_file_path, "wb") as f:
            f.write(file.file.read())  # Use file.file.read() for synchronous reading

        # Step 2: Call your prediction function
        prediction_result = disease.plant_disease_prediction_service.predict(
            local_file_path
        )

        # Extract prediction details
        plant_name = prediction_result["plant_name"]
        disease_name = prediction_result["disease_name"]
        confidence = prediction_result["confidence"] * 100  # Convert to percentage
        if not store:
            # If store is False, return the prediction result without storing
            if os.path.exists(local_file_path):
                os.remove(local_file_path)
            return {
                "plant_name": plant_name,
                "disease_name": disease_name,
                "confidence": confidence,
            }
        # Step 3: Upload the file to Appwrite Storage
        response = STORAGE.create_file(
            bucket_id=BUCKET_DISEASE,  # Replace with your Appwrite bucket ID
            file_id=ID.unique(),  # Generate a unique file ID
            file=InputFile.from_path(
                local_file_path
            ),  # Open the file in read-binary mode
        )
        file_id = response["$id"]

        # Step 4: Store metadata in the Appwrite Collection
        document = DATABASES.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_DISEASE,  # Replace with your collection ID
            document_id=ID.unique(),  # Generate a unique document ID
            data={
                "user_id": user_id,
                "file_id": file_id,
                "plant_name": plant_name,
                "disease_name": disease_name,
                "confidence": confidence,
                "uploaded_at": datetime.now(
                    timezone.utc
                ).isoformat(),  # Store current timestamp
            },
        )

        # Step 5: Delete the local file after processing
        if os.path.exists(local_file_path):
            os.remove(local_file_path)

        # Step 6: Return the response
        return document

    except Exception as e:
        # Ensure the local file is deleted even if an error occurs
        if os.path.exists(local_file_path):
            os.remove(local_file_path)
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@ai_router.get("/disease/history")
def disease_prediction_history(email: Optional[str] = None):
    """
    API endpoint to retrieve the disease prediction history for a user or all users if email is not provided.

    Args:
        email (str, optional): The email of the user. If not provided, returns all entries.

    Returns:
        dict: A list of disease prediction history records.
    """
    try:
        # Step 1: Query the Appwrite database
        queries = []
        if email:
            user_id = get_user_by_email_or_raise(email)["$id"]
            queries.append(Query.equal("user_id", user_id))
        queries.append(Query.limit(10000))
        documents = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_DISEASE,  # Replace with your collection ID
            queries=queries,  # Apply user filter if email is provided
        )

        # Step 2: Extract and return the prediction history
        history = []
        for doc in documents["documents"]:
            file_id = doc["file_id"]
            file_url = get_url(BUCKET_DISEASE, file_id)  # Generate file view URL
            history.append(
                {
                    "file_id": file_id,
                    "plant_name": doc["plant_name"],
                    "disease_name": doc["disease_name"],
                    "confidence": doc["confidence"],
                    "uploaded_at": doc["uploaded_at"],
                    "file_url": file_url,
                }
            )

        # Sort history by uploaded_at in descending order
        history.sort(key=lambda x: x["uploaded_at"])
        return {"history": history}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


def get_weather(lat: float, lon: float, start_date: str, end_date: str):
    try:
        # Validate if start_date is less than end_date
        start_date = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_date, "%Y-%m-%d").date()
        if start_date > end_date:
            raise HTTPException(
                status_code=400, detail="start_date should be less than end_date"
            )

        # Extract the year and month from start_date and end_date
        start_month = start_date.strftime("%m")
        end_month = end_date.strftime("%m")

        # Query the database for existing weather data
        val = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_WEATHER,
            queries=[
                Query.equal("latitude", [lat]),
                Query.equal("longitude", [lon]),
            ],
        )
        # Check if data exists
        if not val["documents"]:
            raise HTTPException(
                status_code=404,
                detail=f"No weather data found for Latitude={lat}, Longitude={lon}",
            )

        # Extract the predictions array
        document = val["documents"][0]  # Assuming one document per lat/lon
        predictions = document["predictions"]  # Stored as an array of JSON strings

        # Filter predictions for the given month range
        filtered_predictions = [
            json.loads(pred)
            for pred in predictions
            if start_month <= json.loads(pred)["month"] <= end_month
        ]
        if not filtered_predictions:
            raise HTTPException(
                status_code=404,
                detail=f"No weather data found for the given month range.",
            )
        # Extract only the required fields: month and other data
        result = [
            {
                "month": pred["month"],  # Use the month directly
                "temperature_2m_max": pred["temperature_2m_max"],
                "temperature_2m_min": pred["temperature_2m_min"],
                "precipitation_sum": pred["precipitation_sum"],
                "wind_speed_10m_max": pred["wind_speed_10m_max"],
                "shortwave_radiation_sum": pred["shortwave_radiation_sum"],
            }
            for pred in filtered_predictions
        ]

        # Sort the results by month in ascending order
        result.sort(key=lambda x: x["month"])

        return result

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching weather data: {str(e)}"
        )


class WeatherPredictionInput(BaseModel):
    """
    Pydantic model for weather prediction input.
    """

    email: str
    end_date: str
    start_date: Optional[str] = None


@ai_router.post("/weather")
def weather_prediction(input_data: WeatherPredictionInput, store: bool = True):
    """
    API endpoint to fetch weather predictions and store them in the database.

    Args:
        input_data (WeatherPredictionInput): The input data containing email, start_date, and end_date.

    Returns:
        list: The weather data for the specified date range.
    """
    try:
        # Step 1: Extract input data
        email = input_data.email
        end_date = input_data.end_date
        start_date = input_data.start_date or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d"
        )

        # Step 2: Get the user ID from the email
        user = get_user_by_email_or_raise(email)
        zipcode = user["zipcode"]

        # Step 3: Fetch coordinates from the database
        coord = get_coord(zipcode)
        lat = coord["latitude"]
        lon = coord["longitude"]

        # Step 4: Call the weather prediction function
        weather_data = get_weather(lat, lon, start_date, end_date)

        # Step 5: Serialize weather_data as an array of JSON strings
        serialized_weather_data = [
            json.dumps(data, default=convert_to_serializable) for data in weather_data
        ]
        if not store:
            # If store is False, return the weather data without storing
            return weather_data
        # Step 6: Store the weather data in the database
        DATABASES.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_WEATHER_HISTORY,  # Replace with your collection ID
            document_id=ID.unique(),
            data={
                "user_id": user["$id"],
                "start_date": datetime.strptime(start_date, "%Y-%m-%d").isoformat(),
                "end_date": datetime.strptime(end_date, "%Y-%m-%d").isoformat(),
                "weather_data": serialized_weather_data,  # Store as an array of JSON strings
                "requested_at": datetime.now(
                    timezone.utc
                ).isoformat(),  # Store current timestamp
            },
        )

        # Step 7: Return the weather data
        return serialized_weather_data

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching weather: {str(e)}")


@ai_router.get("/weather/history")
def weather_prediction_history(email: Optional[str] = None):
    """
    API endpoint to retrieve the weather prediction history for a user or all users if email is not provided.

    Args:
        email (str, optional): The email of the user. If not provided, returns all entries.

    Returns:
        dict: A list of weather prediction history records.
    """
    try:
        # Step 1: Query the Appwrite database
        queries = []
        if email:
            user = get_user_by_email_or_raise(email)
            queries.append(Query.equal("user_id", user["$id"]))
        queries.append(Query.limit(10000))
        documents = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_WEATHER_HISTORY,  # Replace with your collection ID
            queries=queries,  # Apply user filter if email is provided
        )

        # Step 2: Extract and return the prediction history
        history = []
        for doc in documents["documents"]:
            weather_data = [
                json.loads(data) for data in doc["weather_data"]
            ]  # Deserialize JSON strings
            history.append(
                {
                    "start_date": doc["start_date"],
                    "end_date": doc["end_date"],
                    "weather_data": weather_data,
                    "requested_at": doc["requested_at"],
                }
            )

        # Sort history by end_date in descending order
        history.sort(key=lambda x: x["end_date"])
        return {"history": history}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


def fetch_prices(
    lat: float, lon: float, crop: Optional[str], start_date: str, end_date: str
) -> Dict[str, Any]:
    """
    Fetch prices and dates for a given crop, latitude, longitude, and date range.
    If crop is not provided, fetch data for all crops.

    Args:
        lat (float): Latitude of the location.
        lon (float): Longitude of the location.
        crop (Optional[str]): Crop type. UPPERCASE. If None, fetch for all crops.
        start_date (str): Start date in YYYY-MM-DD format.
        end_date (str): End date in YYYY-MM-DD format.

    Returns:
        Dict[str, Any]: A dictionary containing dates and prices for the crop(s).
    """
    try:
        # Build the query
        queries = [
            Query.equal("latitude", [lat]),
            Query.equal("longitude", [lon]),
            Query.limit(10000),
        ]
        if crop:
            crop = crop.upper()
            queries.append(Query.equal("crop", [crop]))

        # Query the PRICES collection
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_PRICES,
            queries=queries,
        )

        if not documents["documents"]:
            raise HTTPException(
                status_code=404,
                detail="No price data found for the given crop and location.",
            )

        # Prepare the result
        result = {}

        for document in documents["documents"]:
            crop_name = document["crop"]
            all_dates = document["dates"]
            all_prices = document["prices"]

            # Filter dates and prices based on the given range
            filtered_dates = []
            filtered_prices = []
            for date, price in zip(all_dates, all_prices):
                if start_date <= date <= end_date:
                    filtered_dates.append(date)
                    filtered_prices.append(price)
            filtered_data = sorted(
                zip(filtered_dates, filtered_prices), key=lambda x: x[0]
            )
            filtered_dates = [item[0] for item in filtered_data]
            filtered_prices = [item[1] for item in filtered_data]
            # Add to the result
            result[crop_name] = {"dates": filtered_dates, "prices": filtered_prices}
        if crop:
            # If a specific crop was requested, return only that crop's data
            if crop not in result:
                raise HTTPException(
                    status_code=404,
                    detail=f"No price data found for the crop '{crop}' in the given date range.",
                )
            return result[crop]
        else:
            return result

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")


@ai_router.post("/prices")
def fetch_prices_api(
    crop_type: str,
    email: str,
    end_date: str,
    start_date: Optional[str] = None,
    store: bool = True,
):
    """
    API to fetch prices for a crop and store the result in PRICES_HISTORY.

    Args:
        crop_type (str): The crop type. UPPERCASE
        email (str): The user's email.
        end_date (str): The end date in YYYY-MM-DD format.
        start_date (str, optional): The start date in YYYY-MM-DD format. Defaults to today.
        store (bool): Whether to store the result in the database. only for internal use.

    Returns:
        dict: The fetched prices and dates.
    """
    try:
        # Set start_date to today if not provided
        start_date = start_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        crop_type = crop_type.upper()
        # Validate date range
        if datetime.strptime(start_date, "%Y-%m-%d") > datetime.strptime(
            end_date, "%Y-%m-%d"
        ):
            raise HTTPException(
                status_code=400,
                detail="start_date should be less than or equal to end_date.",
            )

        # Get user details and coordinates
        user = get_user_by_email_or_raise(email)
        coord = get_coord(user["zipcode"])
        latitude = coord["latitude"]
        longitude = coord["longitude"]

        # Fetch prices using the helper function
        prices_data = fetch_prices(latitude, longitude, crop_type, start_date, end_date)
        if not store:
            # If store is False, return the prices data without storing
            return prices_data
        # Store the result in PRICES_HISTORY
        DATABASES.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_PRICES_HISTORY,
            document_id=ID.unique(),
            data={
                "user_id": user["$id"],
                "dates": prices_data["dates"],
                "prices": prices_data["prices"],
                "requested_at": datetime.now(timezone.utc).isoformat(),
            },
        )

        return prices_data

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching prices: {str(e)}")


@ai_router.get("/prices/history")
def get_prices_history(email: str):
    """
    API to retrieve all price history records for a user.

    Args:
        email (str): The user's email.

    Returns:
        dict: A list of price history records.
    """
    try:
        # Get user details
        user = get_user_by_email_or_raise(email)

        # Query the PRICES_HISTORY collection
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_PRICES_HISTORY,
            queries=[
                Query.equal("user_id", user["$id"]),
                Query.limit(10000),
            ],
        )

        # Sort documents by requested_at in descending order
        history = sorted(
            documents["documents"],
            key=lambda x: x["requested_at"],
        )

        return {"history": history}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching price history: {str(e)}"
        )


def llm(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy LLM function to simulate crop prediction based on input data.

    Args:
        data (Dict[str, Any]): The input data containing soil, weather, and price information.

    Returns:
        Dict[str, Any]: The predicted crop and its details.
    """
    # Simulate LLM processing
    try:
        result = json.dumps(recommendation_api.get_recommendations(data))
        return result
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error during LLM processing: {str(e)}"
        )


def get_available_subsidies(email: str, status: str = "listed"):
    """
    Fetches available subsidies for a farmer and checks if the farmer has already applied for them.

    Args:
        email (str): The email of the farmer.
        status (str): The status of the subsidies to fetch (default is "listed").

    Returns:
        list: A list of available subsidies the farmer has not yet applied for.
    """
    user = get_user_by_email_or_raise(email)
    farmer_id = user["$id"]

    # Fetch subsidies
    a_sub = subsidies.get_subsidies(email, status=status)
    available_subsidies = []

    for subsidy in a_sub["documents"]:
        existing_requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SUBSIDY_REQUESTS,
            queries=[
                Query.equal("farmer_id", [farmer_id]),
                Query.equal("subsidy_id", [subsidy["$id"]]),
                Query.limit(10000),
            ],
        )
        if existing_requests["total"] == 0:
            available_subsidies.append(subsidy)

        # Stop if we have 5 subsidies
        if len(available_subsidies) == 10:
            break

    return available_subsidies


def get_yield(crop: str) -> int:
    """
    Returns the average yield (kg per acre) for the given crop.
    If the crop is unknown, returns a default estimate.
    """
    crop = crop.lower()
    crop_yields = {
        "jowar": 1000,
        "maize": 2500,
        "mango": 6000,
        "onion": 12000,
        "potato": 20000,
        "rice": 2500,
        "wheat": 2500,
    }

    crop = crop.lower()
    if crop in crop_yields:
        return crop_yields[crop]
    else:
        # Default estimate for unknown crops

        return 2000  # Default average yield in kg/acre


class CropPredictionInput(BaseModel):
    """
    Pydantic model for crop prediction input.
    """

    email: str
    end_date: str
    start_date: Optional[str] = None
    acres: int
    soil_type: Optional[str] = None


@ai_router.post("/crop_prediction")
def crop_prediction(
    email: str,
    end_date: str,
    acres: int,
    start_date: Optional[str] = None,
    soil_type: Optional[str] = None,
    file: Optional[UploadFile] = File(None),  # Make file optional
):
    local_file_path = None

    try:
        # Step 1: Validate input data
        start_date = start_date or datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            > datetime.strptime(end_date, "%Y-%m-%d").date()
        ):
            raise HTTPException(
                status_code=400, detail="start_date should be less than end_date"
            )

        # Step 2: Get user details and coordinates
        user = get_user_by_email_or_raise(email)
        user_id = user["$id"]
        cord = get_coord(user["zipcode"])
        latitude = cord["latitude"]
        longitude = cord["longitude"]

        # Step 3: Handle soil classification or file upload
        file_url = None
        soil_data = None
        if file is None and soil_type is None:
            raise HTTPException(
                status_code=400, detail="Either file or soil_type must be provided"
            )

        if soil_type is None:
            # Save the file locally

            local_file_path = f"/tmp/{file.filename}"
            try:
                with open(local_file_path, "wb") as f:
                    f.write(file.file.read())

                # Upload the file to Appwrite storage
                response = STORAGE.create_file(
                    bucket_id=BUCKET_CROP,
                    file_id=ID.unique(),
                    file=InputFile.from_path(local_file_path),
                )
                file_id = response["$id"]
                file_url = get_url(BUCKET_CROP, file_id)

            except Exception as e:
                raise HTTPException(
                    status_code=500, detail=f"Error handling file: {str(e)}"
                )
            finally:
                # Clean up the local file
                if local_file_path and os.path.exists(local_file_path):
                    os.remove(local_file_path)

        else:
            # Use the provided soil_type
            soil_type = soil_type.upper()
            if soil_type not in [
                "BLACK",
                "ALLUVIAL",
                "CINDER",
                "CLAY",
                "LATERITE",
                "LOAMY",
                "PEAT",
                "RED",
                "SANDY",
                "YELLOW",
            ]:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid soil type. Valid options are: black, Alluvial, Cinder, Clay, Laterite, Loamy, Peat, Red, Sandy, Yellow",
                )
            soil_data = {
                "soil_type": soil_type,
                "confidence": 100.0,  # Assuming 100% confidence if provided
            }

        # Step 4: Call the weather prediction API
        weather_input = WeatherPredictionInput(
            email=email, start_date=start_date, end_date=end_date
        )
        weather_predictions = weather_prediction(weather_input, store=False)

        # Step 5: Fetch crop prices
        crop_prices = fetch_prices(
            latitude,
            longitude,
            None,
            start_date,
            end_date,
        )

        applicable = get_available_subsidies(email, status="listed")
        subsidy_ids = [subsidy["$id"] for subsidy in applicable]

        # Step 6: Merge all data into a single hash
        combined_data = {
            "soil_type": soil_data["soil_type"] if file is None else None,
            "soil_type_confidence": soil_data["confidence"] if file is None else None,
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "weather_predictions": weather_predictions,
            "land_size": acres,
            "crops_data": [],
            "subsidies": applicable,
        }

        # Step 7: Process crop prices and contracts
        contract_ids = {}
        available_contracts = contracts.get_contracts(email, "listed")["documents"]
        for crop, value in crop_prices.items():
            # Fetch at most 2 contracts
            contract_for_crop = [
                contract
                for contract in available_contracts
                if contract["crop_type"] == crop
            ][:2]
            contract_ids[crop] = [contract["$id"] for contract in contract_for_crop]

            # Call get_yield function (assume it exists and returns yield per kg)
            yield_per_kg = get_yield(crop)

            # Append crop data
            combined_data["crops_data"].append(
                {
                    "crop_name": crop,
                    "prices": value["prices"],
                    "dates": value["dates"],
                    "contracts": contract_for_crop,
                    "yield_per_kg": yield_per_kg,
                }
            )
        print("\n\n\n\n")
        print(combined_data)
        print("\n\n\n\n")
        # Step 8: Call the LLM function
        llm_result = llm(combined_data)

        # Step 9: Store the input and output in COLLECTION_CROPS
        input_data_to_store = {
            "email": email,
            "start_date": start_date,
            "end_date": end_date,
            "acres": acres,
            "soil_type": soil_type if file is None else file_url,
        }
        DATABASES.create_document(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_CROPS,
            document_id=ID.unique(),
            data={
                "user_id": user_id,
                "requested_at": datetime.now(timezone.utc).isoformat(),
                "input": json.dumps(input_data_to_store),
                "output": llm_result,
            },
        )

        # Step 10: Return the result from the LLM function
        return json.loads(llm_result)

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


@ai_router.get("/crop_prediction/history")
def get_crop_prediction_history(email: str):
    """
    API to retrieve all crop prediction history records for a user.

    Args:
        email (str): The user's email.

    Returns:
        dict: A list of crop prediction history records.
    """
    try:
        # Get user details
        user = get_user_by_email_or_raise(email)
        user_id = user["$id"]

        # Query the COLLECTION_CROPS collection
        documents = DATABASES.list_documents(
            database_id=DATABASE_ID,
            collection_id=COLLECTION_CROPS,
            queries=[
                Query.equal("user_id", user_id),
                Query.limit(10000),
            ],
        )

        # Sort documents by requested_at in descending order
        history = sorted(
            documents["documents"],
            key=lambda x: x["requested_at"],
        )

        return {"history": history}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching crop prediction history: {str(e)}"
        )


def test_price_apis_and_crop_prediction():
    """
    Test the price APIs and crop prediction API.
    """
    try:
        # Test data
        EMAIL = "farmerx@example.com"
        START_DATE = "2025-07-01"
        END_DATE = "2025-11-18"
        SOIL_TYPE = "BLACK"
        data = {
            "email": EMAIL,
            "name": "John Doe",
            "zipcode": "900099",
            "role": "farmer",
            "address": "123 Main St, Springfield",
        }
        user = DATABASES.create_document(
            DATABASE_ID, COLLECTION_USERS, ID.unique(), data
        )
        # # Test fetch_prices_api
        # print("Testing fetch_prices_api...")
        # prices_result = fetch_prices_api(
        #     crop_type=CROP_TYPE,
        #     email=EMAIL,
        #     end_date=END_DATE,
        #     start_date=START_DATE,
        # )
        # print("Fetch Prices API Result:", prices_result)

        # # Test get_prices_history
        # print("Testing get_prices_history...")
        # history_result = get_prices_history(email=EMAIL)
        # print("Get Prices History API Result:", history_result)

        # Test crop_prediction
        print("Testing crop_prediction...")

        crop_prediction_result = crop_prediction(
            email=EMAIL,
            start_date=START_DATE,
            end_date=END_DATE,
            acres=2,
            file=None,  # No file provided for this test
            soil_type=SOIL_TYPE,  # Pass soil type directly
        )
        print("Crop Prediction API Result:", crop_prediction_result)
        DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, user["$id"])
    except Exception as e:
        print(f"Error during testing: {str(e)}")
        DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, user["$id"])


def tests():
    from io import BytesIO

    user = None
    contract1 = None
    subsidy1 = None
    try:
        # Test data
        EMAIL = "farmerx@example.com"
        START_DATE = "2025-07-01"
        END_DATE = "2025-07-18"
        SOIL_IMAGE_PATH = "test_data/soil.jpg"
        DISEASE_IMAGE_PATH = "test_data/crop.jpg"
        data = {
            "email": EMAIL,
            "name": "John Doe",
            "zipcode": "900099",
            "role": "farmer",
            "address": "123 Main St, Springfield",
        }
        user = DATABASES.create_document(
            DATABASE_ID, COLLECTION_USERS, ID.unique(), data
        )
        contract1 = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_CONTRACTS,
            ID.unique(),
            {
                "crop_type": "Wheat",
                "quantity": 100,
                "price_per_kg": 20,
                "advance_payment": 1000,
                "delivery_date": "2025-07-10",
                "payment_terms": "Cash on delivery",
                "locations": ["900099"],
                "status": "listed",
                "buyer_id": "000000000000000000000001",
            },
        )
        subsidy1 = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_SUBSIDIES,
            ID.unique(),
            {
                "program": "Goat Farming Support",
                "description": "Provides goats to farmers for livestock development.",
                "eligibility": "Must own farmland and have livestock experience.",
                "type": "asset",
                "benefits": "2 goats",
                "application_process": "Apply online with proof of farmland ownership.",
                "locations": ["900099", "00000"],
                "dynamic_fields": '{"field1": "value1"}',
                "max_recipients": 50,
                "provider": "Livestock Development Board",
            },
        )

        # Helper function to create an UploadFile object
        def create_upload_file(file_path: str) -> UploadFile:
            with open(file_path, "rb") as f:
                file_bytes = BytesIO(f.read())
            return UploadFile(filename=file_path.split("/")[-1], file=file_bytes)

        # Test soil classification
        print("Testing soil_classification...")
        soil_file = create_upload_file(SOIL_IMAGE_PATH)
        soil_result = soil_classification(email=EMAIL, file=soil_file, store=True)
        print("Soil Classification Result:", soil_result)

        # Test disease prediction
        print("Testing disease_prediction...")
        disease_file = create_upload_file(DISEASE_IMAGE_PATH)
        disease_result = disease_prediction(email=EMAIL, file=disease_file, store=True)
        print("Disease Prediction Result:", disease_result)

        # Test weather prediction
        print("Testing weather_prediction...")
        weather_input = WeatherPredictionInput(
            email=EMAIL, start_date=START_DATE, end_date=END_DATE
        )
        weather_result = weather_prediction(input_data=weather_input, store=True)
        print("Weather Prediction Result:", weather_result)

        # Test crop prediction
        print("Testing crop_prediction...")
        crop_file = create_upload_file(SOIL_IMAGE_PATH)
        crop_input = CropPredictionInput(
            email=EMAIL, start_date=START_DATE, end_date=END_DATE, acres=2
        )
        crop_result = crop_prediction(input_data=crop_input, file=crop_file)
        print("Crop Prediction Result:", crop_result)

    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        # Cleanup: Delete the test user
        try:
            DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, user["$id"])
            print("Deleted test user.")
        except Exception as e:
            print(f"Error deleting test user: {str(e)}")

        # Cleanup: Delete soil classification entries
        try:
            soil_documents = DATABASES.list_documents(
                DATABASE_ID,
                COLLECTION_SOIL,
                queries=[Query.equal("user_id", user["$id"])],
            )
            for doc in soil_documents["documents"]:
                DATABASES.delete_document(DATABASE_ID, COLLECTION_SOIL, doc["$id"])
                STORAGE.delete_file(BUCKET_SOIL, doc["file_id"])
            print("Deleted soil classification entries.")
        except Exception as e:
            print(f"Error deleting soil classification entries: {str(e)}")

        # Cleanup: Delete disease prediction entries
        try:
            disease_documents = DATABASES.list_documents(
                DATABASE_ID,
                COLLECTION_DISEASE,
                queries=[Query.equal("user_id", user["$id"])],
            )
            for doc in disease_documents["documents"]:
                DATABASES.delete_document(DATABASE_ID, COLLECTION_DISEASE, doc["$id"])
                STORAGE.delete_file(BUCKET_DISEASE, doc["file_id"])
            print("Deleted disease prediction entries.")
        except Exception as e:
            print(f"Error deleting disease prediction entries: {str(e)}")

        # Cleanup: Delete weather prediction entries
        try:
            weather_documents = DATABASES.list_documents(
                DATABASE_ID,
                COLLECTION_WEATHER_HISTORY,
                queries=[Query.equal("user_id", user["$id"])],
            )
            for doc in weather_documents["documents"]:
                DATABASES.delete_document(
                    DATABASE_ID, COLLECTION_WEATHER_HISTORY, doc["$id"]
                )
            # delete contract1 and subsidy1
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_CONTRACTS, contract1["$id"]
            )
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_SUBSIDIES, subsidy1["$id"]
            )
            print("Deleted weather prediction entries.")
        except Exception as e:
            print(f"Error deleting weather prediction entries: {str(e)}")


def test_test():
    data = {
        "email": "farmerx@example.com",
        "name": "John Doe",
        "zipcode": "900099",
        "role": "farmer",
        "address": "123 Main St, Springfield",
    }
    user = DATABASES.create_document(DATABASE_ID, COLLECTION_USERS, ID.unique(), data)
    start_date = datetime.now().strftime("%Y-%m-%d")
    end_date = (datetime.now() + timedelta(days=9 * 30)).strftime("%Y-%m-%d")

    print(
        weather_prediction(
            WeatherPredictionInput(
                email="farmerx@example.com",
                start_date=start_date,
                end_date=end_date,
            )
        )
    )
    DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, user["$id"])
