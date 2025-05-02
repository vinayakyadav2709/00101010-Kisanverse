

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
    "date": "2025-07-01",
    "temperature": 30.5,
    "humidity": 70.0,
    "rainfall": 5.0
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
          "date": "2025-07-01",
          "temperature": 30.5,
          "humidity": 70.0,
          "rainfall": 5.0
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

### **Response**
- **Success**: Returns the predicted crops and their details. dates in ascending order.
- **Error**: Returns an error message if the prediction fails.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/predictions/crop_prediction \
-H "Content-Type: application/json" \
-d '{
  "email": "farmer@example.com",
  "start_date": "2025-07-01",
  "end_date": "2025-07-10",
  "acres": 5
}'
```

### **Example Response**
```json
{
  "soil_type": "Loamy",
  "soil_type_confidence": 95.0,
  "latitude": 28.7041,
  "longitude": 77.1025,
  "start_date": "2025-07-01",
  "end_date": "2025-07-10",
  "weather_predictions": [{
          "date": "2025-07-01",
          "temperature": 30.5,
          "humidity": 70.0,
          "rainfall": 5.0
        },...],
  "land_size": 5,
  "crops_data": [
    {
      "crop_name": "Wheat",
      "prices": [100, 105, 110],
      "dates": ["2025-07-01", "2025-07-02", "2025-07-03",...],
      "contracts": ["id"],
      "yield_per_kg": 2500
    },
    ...
  ],
  "subsidies": ["id"]
}
```
