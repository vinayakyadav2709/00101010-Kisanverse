from fastapi import APIRouter, HTTPException
from matplotlib.style import available
from pydantic import BaseModel, Field
from typing import Dict, Any
import requests
from fastapi import APIRouter, HTTPException, UploadFile, File
from appwrite.id import ID
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
)
from routers import contracts, subsidies
from core.dependencies import get_user_by_email_or_raise, get_coord
from typing import List, Optional
from pydantic import BaseModel
from appwrite.query import Query
from datetime import datetime, timezone, timedelta
from models import disease, soil, weather

# import price
import json
from torchvision import transforms
from PIL import Image
import torch

import os

# Define the router
ai_router = APIRouter(prefix="/predictions", tags=["Predictions"])
import numpy as np  # Add this import if not already present


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

        documents = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SOIL,  # Replace with your collection ID
            queries=queries,  # Apply user filter if email is provided
        )

        # Step 2: Extract and return the history
        history = []
        for doc in documents["documents"]:
            file_id = doc["file_id"]
            file_url = f"http://localhost/v1/storage/buckets/{BUCKET_SOIL}/files/{file_id}/view"  # Generate file view URL
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
        history.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return {"history": history}

    except Exception as e:
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

        documents = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_DISEASE,  # Replace with your collection ID
            queries=queries,  # Apply user filter if email is provided
        )

        # Step 2: Extract and return the prediction history
        history = []
        for doc in documents["documents"]:
            file_id = doc["file_id"]
            file_url = STORAGE.get_file_view(BUCKET_DISEASE, file_id)[
                "href"
            ]  # Generate file view URL
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
        history.sort(key=lambda x: x["uploaded_at"], reverse=True)
        return {"history": history}

    except Exception as e:
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

        # Adjust start_date and end_date to include one month before and after
        adjusted_start_date = (start_date - timedelta(days=30)).strftime("%Y-%m-%d")
        adjusted_end_date = (end_date + timedelta(days=30)).strftime("%Y-%m-%d")

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

        # Filter predictions for the adjusted date range
        filtered_predictions = [
            json.loads(pred)
            for pred in predictions
            if adjusted_start_date <= json.loads(pred)["date"] <= adjusted_end_date
        ]

        # Extract only the required fields: month and other data (excluding date)
        result = [
            {
                "month": json.loads(pred)["month"].split("-")[
                    1
                ],  # Extract only the month
                "temperature_2m_max": json.loads(pred)["temperature_2m_max"],
                "temperature_2m_min": json.loads(pred)["temperature_2m_min"],
                "precipitation_sum": json.loads(pred)["precipitation_sum"],
                "wind_speed_10m_max": json.loads(pred)["wind_speed_10m_max"],
                "shortwave_radiation_sum": json.loads(pred)["shortwave_radiation_sum"],
            }
            for pred in filtered_predictions
        ]

        # Sort the results by month in ascending order
        result.sort(key=lambda x: int(x["month"]))

        return result

    except Exception as e:
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
        return weather_data

    except Exception as e:
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
        history.sort(key=lambda x: x["end_date"], reverse=True)
        return {"history": history}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching history: {str(e)}")


def get_prices(start_date: str, end_date: str, lat: float, lon: float):
    # dummy
    return {
        "wheat": {
            "dates": [
                "2023-10-01",
                "2023-10-02",
                "2023-10-03",
            ],
            "prices": [100, 105, 110],
        },
        "rice": {
            "dates": [
                "2023-10-01",
                "2023-10-02",
                "2023-10-03",
            ],
            "prices": [200, 205, 210],
        },
    }


def llm(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Dummy LLM function to simulate crop prediction based on input data.

    Args:
        data (Dict[str, Any]): The input data containing soil, weather, and price information.

    Returns:
        Dict[str, Any]: The predicted crop and its details.
    """
    # Simulate LLM processing
    data["sample"] = "sample"
    return data


class CropPredictionInput(BaseModel):
    """
    Pydantic model for crop prediction input.
    """

    email: str
    end_date: str
    start_date: Optional[str] = None
    acres: int


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


@ai_router.post("/crop_prediction")
def crop_prediction(
    input_data: CropPredictionInput,
    file: Optional[UploadFile] = File(None),  # Make file optional
    soil_type: Optional[str] = None,
):
    try:
        # Step 1: Extract input data
        email = input_data.email
        end_date = input_data.end_date
        start_date = input_data.start_date or datetime.now(timezone.utc).strftime(
            "%Y-%m-%d"
        )
        # validate if start_date is less than end_date
        if (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            > datetime.strptime(end_date, "%Y-%m-%d").date()
        ):
            raise HTTPException(
                status_code=400, detail="start_date should be less than end_date"
            )
        user = get_user_by_email_or_raise(email)
        cord = get_coord(user["zipcode"])
        latitude = cord["latitude"]
        longitude = cord["longitude"]

        # Step 2: Call the soil classification API as a function
        if file is None and soil_type is None:
            raise HTTPException(
                status_code=400, detail="Either file or soil_type must be provided"
            )
        if soil_type is None:
            soil_data = soil_classification(email=email, file=file, store=False)
        else:
            soil_data = {
                "soil_type": soil_type,
                "confidence": 100.0,  # Assuming 100% confidence if provided
            }
        # Step 3: Call the weather prediction API as a function
        weather_input = WeatherPredictionInput(
            email=email, start_date=start_date, end_date=end_date
        )
        weather_predictions = weather_prediction(weather_input, store=False)

        # Step 4: Fetch crop prices
        crop_prices = get_prices(start_date, end_date, latitude, longitude)
        applicable = get_available_subsidies(email, status="listed")
        # extract subsidies id
        subsidy_ids = [subsidy["$id"] for subsidy in applicable]
        # Step 5: Merge all data into a single hash
        combined_data = {
            "soil_type": soil_data["soil_type"],
            "soil_type_confidence": soil_data["confidence"],
            "latitude": latitude,
            "longitude": longitude,
            "start_date": start_date,
            "end_date": end_date,
            "weather_predictions": weather_predictions,
            "land_size": input_data.acres,
            "crops_data": [],
            "subsidies": applicable,
        }
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

        # Step 6: Call the LLM function
        llm_result = llm(combined_data)

        # Step 7: Return the result from the LLM function
        return llm_result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


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
