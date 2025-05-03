

# API Documentation for `/predictions` Endpoint

This document provides details about the API endpoints defined in the `ai.py` file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Soil Classification**

### **Endpoint**
`POST /predictions/soil_type`

### **Description**
Classifies the soil type based on the uploaded image.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string): The email of the user.
- `store` (boolean, optional): Whether to store the result in the database. Defaults to `true`.

### **Request Body**
- `file` (file): The soil image file to classify.

### **Response**
- **Success**: Returns the soil type and confidence level.
- **Error**: Returns an error message if the classification fails.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/predictions/soil_type?email=farmer@example.com \
-H "Content-Type: multipart/form-data"   \
-F "file=@soil.jpg"
```

### **Example Response**
```json
{
  "soil_type": "Loamy",
  "confidence": 95.0
}
```

---

## **2. Soil Classification History**

### **Endpoint**
`GET /predictions/soil_type/history`

### **Description**
Fetches the soil classification history for a user or all users.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string, optional): The email of the user. If not provided, returns all entries.

### **Response**
- **Success**: Returns a list of soil classification history records.
- **Error**: Returns an error message if the history retrieval fails.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/predictions/soil_type/history?email=farmer@example.com
```

### **Example Response**
```json
{
  "history": [
    {
      "file_id": "file_id",
      "file_url": "http://localhost/v1/storage/buckets/soil/files/file_id/view",
      "soil_type": "Loamy",
      "confidence": 95.0,
      "uploaded_at": "2025-05-01T12:00:00.000+00:00"
    }
  ]
}
```

---

## **3. Disease Prediction**

### **Endpoint**
`POST /predictions/disease`

### **Description**
Predicts plant diseases based on the uploaded image.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string): The email of the user.
- `store` (boolean, optional): Whether to store the result in the database. Defaults to `true`.

### **Request Body**
- `file` (file): The plant image file to analyze.

### **Response**
- **Success**: Returns the plant name, disease name, and confidence level.
- **Error**: Returns an error message if the prediction fails.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/predictions/disease?email=farmer@example.com \
-H "Content-Type: multipart/form-data"   \
-F "file=@plant.jpg"
```

### **Example Response**
```json
{
  "plant_name": "Tomato",
  "disease_name": "Late Blight",
  "confidence": 92.5
}
```

---

## **4. Disease Prediction History**

### **Endpoint**
`GET /predictions/disease/history`

### **Description**
Fetches the disease prediction history for a user or all users.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string, optional): The email of the user. If not provided, returns all entries.

### **Response**
- **Success**: Returns a list of disease prediction history records.
- **Error**: Returns an error message if the history retrieval fails.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/predictions/disease/history?email=farmer@example.com
```

### **Example Response**
```json
{
  "history": [
    {
      "file_id": "file_id",
      "file_url": "http://localhost/v1/storage/buckets/disease/files/file_id/view",
      "plant_name": "Tomato",
      "disease_name": "Late Blight",
      "confidence": 92.5,
      "uploaded_at": "2025-05-01T12:00:00.000+00:00"
    }
  ]
}
```

---

## **5. Weather Prediction**

### **Endpoint**
`POST /predictions/weather`

### **Description**
Fetches weather predictions for a specific date range based on the user's location.

### **Who Can Call It**
Anyone.

### **Request Body**
```json
{
  "email": "string",
  "start_date": "string (optional, YYYY-MM-DD)",
  "end_date": "string (YYYY-MM-DD)"
}
```

### **Response**
- **Success**: Returns the weather data for the specified date range. ascending order dates
- **Error**: Returns an error message if the prediction fails.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/predictions/weather \
-H "Content-Type: application/json" \
-d '{
  "email": "farmer@example.com",
  "start_date": "2025-07-01",
  "end_date": "2025-07-10"
}'
```

### **Example Response**
```json
[
    {
        "month": "04",
        "temperature_2m_max": 30.5,
        "temperature_2m_min": 29.4,
        "precipitation_sum": 0.0,
        "wind_speed_10m_max": 22.5,
        "shortwave_radiation_sum": 25.26
    },
    {
        "month": "05",
        "temperature_2m_max": 32.1,
        "temperature_2m_min": 30.0,
        "precipitation_sum": 1.2,
        "wind_speed_10m_max": 20.0,
        "shortwave_radiation_sum": 24.50
    },
    ...
]
```

---

## **6. Weather Prediction History**

### **Endpoint**
`GET /predictions/weather/history`

### **Description**
Fetches the weather prediction history for a user or all users.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string, optional): The email of the user. If not provided, returns all entries.

### **Response**
- **Success**: Returns a list of weather prediction history records. ascending order dates
- **Error**: Returns an error message if the history retrieval fails.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/predictions/weather/history?email=farmer@example.com
```

### **Example Response**
```json
{
  "history": [
    {
      "start_date": "2025-07-01",
      "end_date": "2025-07-10",
      "weather_data": [
          {
              "month": "04",
              "temperature_2m_max": 30.5,
              "temperature_2m_min": 29.4,
              "precipitation_sum": 0.0,
              "wind_speed_10m_max": 22.5,
              "shortwave_radiation_sum": 25.26
          },
          {
              "month": "05",
              "temperature_2m_max": 32.1,
              "temperature_2m_min": 30.0,
              "precipitation_sum": 1.2,
              "wind_speed_10m_max": 20.0,
              "shortwave_radiation_sum": 24.50
          },
        ...
        ],
      "requested_at": "2025-05-01T12:00:00.000+00:00"
    }
  ]
}
```

---

## **7. Crop Prediction**

### **Endpoint**
`POST /predictions/crop_prediction`

### **Description**
Predicts the best crops to grow based on soil, weather, and price data.

### **Who Can Call It**
Anyone.

### **Request Body**
```json
{
  "email": "string",
  "start_date": "string (optional, YYYY-MM-DD)",
  "end_date": "string (YYYY-MM-DD)",
  "acres": "integer"
}
```

### **Query Parameters**
- `file` (file, optional): The soil image file for classification.
- `soil_type` (string, optional): The soil type if known.
- Either `file` or `soil_type` needs to be provided. If both are given, `soil_type` is used.

### **Response**
- **Success**: Returns the predicted crops and their details. Dates are in ascending order.
- **Error**: Returns an error message if the prediction fails.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/predictions/crop_prediction \
-F "file=@path/to/soil_image.jpg" \
-F "email=farmer@example.com" \
-F "start_date=2025-07-01" \
-F "end_date=2025-07-10" \
-F "acres=5"
```
If you want to use the `soil_type` parameter instead of uploading a file:
```bash
curl -X POST http://localhost:8000/predictions/crop_prediction \
-H "Content-Type: application/json" \
-d '{
  "email": "farmer@example.com",
  "start_date": "2025-07-01",
  "end_date": "2025-07-10",
  "acres": 5,
  "soil_type": "Loamy"
}'
```
### **Notes**
- If both `file` and `soil_type` are provided, the `soil_type` parameter will take precedence, and the file will not be processed.
- The `file` parameter should be a valid image file (e.g., `.jpg`, `.png`).
- The `start_date` is optional. If not provided, it defaults to the current date.

---

### **Example Response**
```json
{
  "soil_type": "Loamy",
  "soil_type_confidence": 95.0,
  "latitude": 28.7041,
  "longitude": 77.1025,
  "start_date": "2025-07-01",
  "end_date": "2025-07-10",
  "weather_predictions": [
        {
            "month": "04",
            "temperature_2m_max": 30.5,
            "temperature_2m_min": 29.4,
            "precipitation_sum": 0.0,
            "wind_speed_10m_max": 22.5,
            "shortwave_radiation_sum": 25.26
        },
        {
            "month": "05",
            "temperature_2m_max": 32.1,
            "temperature_2m_min": 30.0,
            "precipitation_sum": 1.2,
            "wind_speed_10m_max": 20.0,
            "shortwave_radiation_sum": 24.50
        }
    ],
  "land_size": 5,
  "crops_data": [
    {
      "crop_name": "Wheat",
      "prices": [100, 105, 110],
      "dates": ["2025-07-01", "2025-07-02", "2025-07-03"],
      "contracts": ["id"],
      "yield_per_kg": 2500
    }
  ],
  "subsidies": ["id"]
}
```

---




## **8. Fetch Crop Prices**

### **Endpoint**
`POST /predictions/prices`

### **Description**
Fetches crop prices for a specific crop and date range based on the user's location. Optionally stores the result in the `PRICES_HISTORY` collection.

### **Who Can Call It**
Internal and external users.

### **Query Parameters**
- `crop_type` (string): The crop type. Must be in uppercase.
- `email` (string): The email of the user.
- `end_date` (string): The end date in `YYYY-MM-DD` format.
- `start_date` (string, optional): The start date in `YYYY-MM-DD` format. Defaults to today.
- `store` (boolean, optional): Whether to store the result in the database. Defaults to `true`. For internal use only.

### **Response**
- **Success**: Returns the crop prices and dates.
- **Error**: Returns an error message if the price retrieval fails.

### **Example API Call**
```bash
curl -X POST "http://localhost:8000/predictions/prices?crop_type=WHEAT&email=farmer@example.com&start_date=2025-07-01&end_date=2025-07-10&store=true"
```

### **Example Response**
```json
{
  "dates": ["2025-07-01", "2025-07-02", "2025-07-03"],
  "prices": [100, 105, 110]
}
```

---

## **9. Fetch Price History**

### **Endpoint**
`GET /predictions/prices/history`

### **Description**
Fetches the price history for a user from the `PRICES_HISTORY` collection.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string): The email of the user.

### **Response**
- **Success**: Returns a list of price history records sorted by `requested_at` in descending order.
- **Error**: Returns an error message if the history retrieval fails.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/predictions/prices/history?email=farmer@example.com
```

### **Example Response**
```json
{
  "history": [
    {
      "user_id": "user_id",
      "dates": ["2025-07-01", "2025-07-02", "2025-07-03"],
      "prices": [100, 105, 110],
      "requested_at": "2025-05-01T12:00:00.000+00:00"
    },
    {
      "user_id": "user_id",
      "dates": ["2025-06-01", "2025-06-02", "2025-06-03"],
      "prices": [90, 95, 100],
      "requested_at": "2025-04-01T12:00:00.000+00:00"
    }
  ]
}
```


---

## **10. Crop Prediction History**

### **Endpoint**
`GET /predictions/crop_prediction/history`

### **Description**
Fetches the crop prediction history for a user.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `email` (string): The email of the user.

### **Response**
- **Success**: Returns a list of crop prediction history records sorted by `requested_at` in descending order.
- **Error**: Returns an error message if the history retrieval fails.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/predictions/crop_prediction/history?email=farmer@example.com
```

### **Example Response**
```json
{
  "history": [
    {
      "user_id": "user_id",
      "requested_at": "2025-05-01T12:00:00.000+00:00",
      "input": {
        "email": "farmer@example.com",
        "start_date": "2025-07-01",
        "end_date": "2025-07-18",
        "acres": 5,
        "soil_type": "http://localhost/v1/storage/buckets/bucket_id/files/file_id/view"
      },
      "output": {
        "recommendations": [
          {
            "rank": 1,
            "crop_name": "Soybean",
            "recommendation_score": 85.5
          }
        ]
      }
    },
    {
      "user_id": "user_id",
      "requested_at": "2025-04-01T12:00:00.000+00:00",
      "input": {
        "email": "farmer@example.com",
        "start_date": "2025-06-01",
        "end_date": "2025-06-15",
        "acres": 3,
        "soil_type": "Loamy"
      },
      "output": {
        "recommendations": [
          {
            "rank": 1,
            "crop_name": "Maize",
            "recommendation_score": 78.0
          }
        ]
      }
    }
  ]
}
```