# API Documentation for `/subsidies` and `/subsidy_requests` Endpoints

This document provides details about the API endpoints defined in the subsidies.py file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Create a Subsidy**

### **Endpoint**
`POST /subsidies`

### **Description**
Creates a new subsidy.

### **Who Can Call It**
- Admins only.

### **Request Body**
- `type` (string, required): The type of the subsidy (`crop`, `seed`, `fertilizer`, `machine`, `general`).
- `locations` (list of strings, required): The locations where the subsidy is applicable.
- `max_recipients` (integer, required): The maximum number of recipients for the subsidy.
- `dynamic_fields` (string, required): Additional dynamic fields in JSON format.
- `provider` (string, required): The name of the organization providing the subsidy.

### **Response**
- **Success**: Returns the created subsidy document.
- **Error**: Returns an error message if the user is not an admin or if the request body is invalid.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/subsidies \
-H "Content-Type: application/json" \
-d '{
  "type": "crop",
  "locations": ["location1", "location2"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "provider": "Organization A"
}'
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["location1", "location2"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "provider": "Organization A",
  "status": "listed",
  "$id": "subsidy_id",
  "$createdAt": "2025-05-01T12:00:00.000+00:00"
}
```

---

## **2. Get Subsidies**

### **Endpoint**
`GET /subsidies`

### **Description**
Fetches subsidies based on optional filters.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `locations` (string or list of strings, optional): Filter subsidies by applicable locations.
- `type` (string, optional): Filter subsidies by type (`crop`, `seed`, `fertilizer`, `machine`, `general`, or `all`).
- `status` (string, optional): Filter subsidies by status (`listed`, `removed`, `fulfilled`, or `all`).
- `provider` (string, optional): Filter subsidies by the provider organization.

### **Response**
- **Success**: Returns a list of subsidies matching the filters.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/subsidies?locations=location1&type=crop&status=listed&provider=Organization%20A"
```

### **Example Response**
```json
{
  "total": 1,
  "documents": [
    {
      "type": "crop",
      "locations": ["location1", "location2"],
      "max_recipients": 100,
      "dynamic_fields": "{\"field1\": \"value1\"}",
      "provider": "Organization A",
      "status": "listed",
      "$id": "subsidy_id",
      "$createdAt": "2025-05-01T12:00:00.000+00:00"
    }
  ]
}
```

---

## **3. Update a Subsidy**

### **Endpoint**
`PATCH /subsidies/{subsidy_id}`

### **Description**
Updates an existing subsidy.

### **Who Can Call It**
- Admins only.

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to update.

### **Request Body**
- Any combination of the following fields:
  - `type` (string, optional): The type of the subsidy.
  - `locations` (list of strings, optional): The locations where the subsidy is applicable.
  - `max_recipients` (integer, optional): The maximum number of recipients for the subsidy.
  - `dynamic_fields` (string, optional): Additional dynamic fields in JSON format.
  - `provider` (string, optional): The name of the organization providing the subsidy.

### **Response**
- **Success**: Returns the updated subsidy document.
- **Error**: Returns an error message if the user is not an admin or if the update is invalid.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidies/subsidy_id \
-H "Content-Type: application/json" \
-d '{
  "max_recipients": 150,
  "locations": ["location3"]
}'
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["location3"],
  "max_recipients": 150,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "provider": "Organization A",
  "status": "listed",
  "$id": "subsidy_id",
  "$updatedAt": "2025-05-01T12:30:00.000+00:00"
}
```

---

## **4. Delete a Subsidy**

### **Endpoint**
`DELETE /subsidies/{subsidy_id}`

### **Description**
Marks a subsidy as removed and rejects all pending requests for it.

### **Who Can Call It**
- Admins only.

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to delete.

### **Response**
- **Success**: Returns the updated subsidy document with the status set to `"removed"`.
- **Error**: Returns an error message if the user is not an admin or if the subsidy does not exist.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/subsidies/subsidy_id
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["location1", "location2"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "provider": "Organization A",
  "status": "removed",
  "$id": "subsidy_id",
  "$updatedAt": "2025-05-01T12:45:00.000+00:00"
}
```

---

## **5. Create a Subsidy Request**

### **Endpoint**
`POST /subsidy_requests`

### **Description**
Creates a new request for a subsidy.

### **Who Can Call It**
- Farmers only.

### **Request Body**
- `subsidy_id` (string, required): The ID of the subsidy being requested.

### **Response**
- **Success**: Returns the created subsidy request document.
- **Error**: Returns an error message if the user is not a farmer, if the subsidy is not listed, or if the farmer has already requested the subsidy.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/subsidy_requests \
-H "Content-Type: application/json" \
-d '{
  "subsidy_id": "subsidy_id"
}'
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "requested",
  "$id": "request_id",
  "$createdAt": "2025-05-01T13:00:00.000+00:00"
}
```

---

## **6. Accept a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/accept`

### **Description**
Accepts a subsidy request.

### **Who Can Call It**
- Admins only.

### **Path Parameters**
- `request_id` (string): The ID of the subsidy request to accept.

### **Response**
- **Success**: Returns the updated subsidy and request documents.
- **Error**: Returns an error message if the user is not an admin or if the request is not in the `"requested"` status.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/accept
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "recipients_accepted": 1,
  "status": "listed",
  "$id": "subsidy_id",
  "$updatedAt": "2025-05-01T13:15:00.000+00:00"
}
```

---

## **7. Reject a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/reject`

### **Description**
Rejects a subsidy request.

### **Who Can Call It**
- Admins only.

### **Path Parameters**
- `request_id` (string): The ID of the subsidy request to reject.

### **Response**
- **Success**: Returns the updated request document.
- **Error**: Returns an error message if the user is not an admin or if the request is not in the `"requested"` status.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/reject
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "rejected",
  "$id": "request_id",
  "$updatedAt": "2025-05-01T13:30:00.000+00:00"
}
```

---

## **8. Delete a Subsidy Request**

### **Endpoint**
`DELETE /subsidy_requests/{request_id}`

### **Description**
Deletes a subsidy request. Farmers can withdraw their own requests, and admins can remove any request.

### **Who Can Call It**
- Farmers: Can withdraw their own requests.
- Admins: Can remove any request.

### **Path Parameters**
- `request_id` (string): The ID of the subsidy request to delete.

### **Response**
- **Success**: Returns the updated request document with the status set to `"withdrawn"` (for farmers) or `"removed"` (for admins).
- **Error**: Returns an error message if the user is not authorized or if the request is not in the `"requested"` or `"accepted"` status.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/subsidy_requests/request_id
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "withdrawn",
  "$id": "request_id",
  "$updatedAt": "2025-05-01T13:45:00.000+00:00"
}
```

--- 

Here is the updated documentation including the `GET /subsidy_requests` and `PATCH /subsidy_requests/{request_id}/fulfill` endpoints:

---

## **9. Get Subsidy Requests**

### **Endpoint**
`GET /subsidy_requests`

### **Description**
Fetches subsidy requests based on optional filters.

### **Who Can Call It**
- Farmers: Can fetch their own requests.
- Admins: Can fetch all requests.

### **Query Parameters**
- `email` (string, optional): The email of the user (used to filter requests for farmers).
- `status` (string, optional): Filter requests by status (`requested`, `accepted`, `rejected`, `withdrawn`, `removed`, `fulfilled`, or `all`).
- `subsidy_id` (string, optional): Filter requests by subsidy ID.

### **Response**
- **Success**: Returns a list of subsidy requests matching the filters.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/subsidy_requests?email=farmer@example.com&status=requested&subsidy_id=subsidy_id"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "subsidy_id": "subsidy_id",
      "farmer_id": "farmer_id",
      "status": "requested",
      "$id": "request_id_1",
      "$createdAt": "2025-05-01T14:00:00.000+00:00"
    },
    {
      "subsidy_id": "subsidy_id",
      "farmer_id": "farmer_id",
      "status": "accepted",
      "$id": "request_id_2",
      "$createdAt": "2025-05-01T14:10:00.000+00:00"
    }
  ]
}
```

---

## **10. Fulfill a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/fulfill`

### **Description**
Marks a subsidy request as fulfilled.

### **Who Can Call It**
- Farmers: Can fulfill their own accepted requests.
- Admins: Can fulfill any accepted request.

### **Path Parameters**
- `request_id` (string): The ID of the subsidy request to fulfill.

### **Response**
- **Success**: Returns the updated request document with the status set to `"fulfilled"`.
- **Error**: Returns an error message if the user is not authorized or if the request is not in the `"accepted"` status.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/fulfill \
-H "Content-Type: application/json" \
-d '{
  "email": "farmer@example.com"
}'
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "fulfilled",
  "$id": "request_id",
  "$updatedAt": "2025-05-01T14:30:00.000+00:00"
}
```

---

Let me know if you need further adjustments or additional details!