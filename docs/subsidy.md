# API Documentation for `/subsidies` and `/subsidy_requests` Endpoints

This document provides details about the API endpoints defined in the subsidies.py file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Create a Subsidy**

### **Endpoint**
`POST /subsidies`

### **Description**
Creates a new subsidy. Only providers can create subsidies.

### **Who Can Call It**
Providers.

### **Request Body**
```json
{
  "type": "string (crop, seed, fertilizer, machine, general)",
  "locations": ["string"],
  "max_recipients": "integer",
  "dynamic_fields": "string (valid JSON)"
}
```

### **Query Parameters**
- `email` (string): The email of the provider creating the subsidy.

### **Response**
- **Success**: Returns the created subsidy document.
- **Error**: Returns an error message if the user is not a provider or if the input is invalid.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/subsidies?email=provider@example.com \
-H "Content-Type: application/json" \
-d '{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}"
}'
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "submitted_by": "provider_id",
  "status": "pending",
  "$id": "subsidy_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **2. Get Subsidies**

### **Endpoint**
`GET /subsidies`

### **Description**
Fetches subsidies based on the user's role and optional filters.

### **Who Can Call It**
- Providers: Fetch subsidies they created.
- Farmers: Fetch approved subsidies applicable to their location.
- Admins: Fetch all subsidies.

### **Query Parameters**
- `email` (string): The email of the user fetching subsidies.
- `type` (string, optional): Filter by subsidy type (`crop`, `seed`, etc.) or status (`pending`, `approved`, etc.).

### **Response**
- **Success**: Returns a list of subsidies.
- **Error**: Returns an error message if the type or status is invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/subsidies?email=farmer@example.com&type=approved"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "type": "crop",
      "locations": ["Haryana", "Punjab"],
      "max_recipients": 100,
      "dynamic_fields": "{\"field1\": \"value1\"}",
      "submitted_by": "provider_id",
      "status": "approved",
      "$id": "subsidy_id_1",
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **3. Update a Subsidy**

### **Endpoint**
`PATCH /subsidies/{subsidy_id}`

### **Description**
Updates a subsidy. Only admins can update subsidies.

### **Who Can Call It**
Admins.

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to update. can only be done if status of subsidy is pending or approved

### **Request Body**
```json
{
  "type": "string (optional)",
  "locations": ["string (optional)"],
  "max_recipients": "integer (optional)",
  "dynamic_fields": "string (optional, valid JSON)",
}
```

### **Response**
- **Success**: Returns the updated subsidy document.
- **Error**: Returns an error message if the update is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidies/subsidy_id \
-H "Content-Type: application/json" \
-d '{
  "max_recipients": 150,
  "dynamic_fields": "{\"field2\": \"value2\"}"
}'
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 150,
  "dynamic_fields": "{\"field2\": \"value2\"}",
  "submitted_by": "provider_id",
  "status": "approved",
  "$id": "subsidy_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **4. Approve a Subsidy**

### **Endpoint**
`PATCH /subsidies/{subsidy_id}/approve`

### **Description**
Approves a pending subsidy. Only admins can approve subsidies.

### **Who Can Call It**
Admins.

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to approve.

### **Response**
- **Success**: Returns the approved subsidy document.
- **Error**: Returns an error message if the subsidy is not in a pending state.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidies/subsidy_id/approve?email=admin@example.com
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "submitted_by": "provider_id",
  "status": "approved",
  "$id": "subsidy_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **5. Reject a Subsidy**

### **Endpoint**
`PATCH /subsidies/{subsidy_id}/reject`

### **Description**
Rejects a pending subsidy. Only admins can reject subsidies.

### **Who Can Call It**
Admins.

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to reject.

### **Response**
- **Success**: Returns the rejected subsidy document.
- **Error**: Returns an error message if the subsidy is not in a pending state.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidies/subsidy_id/reject?email=admin@example.com
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "submitted_by": "provider_id",
  "status": "rejected",
  "$id": "subsidy_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **6. Delete a Subsidy**

### **Endpoint**
`DELETE /subsidies/{subsidy_id}`

### **Description**
Deletes a subsidy. Admins can mark it as "removed," and providers can mark it as "withdrawn."
admin can delete approved and pending ones, provider can only do pending ones.

### **Who Can Call It**
- Admins.
- Providers (only for subsidies they created).

### **Path Parameters**
- `subsidy_id` (string): The ID of the subsidy to delete.

### **Response**
- **Success**: Returns the updated subsidy document with the new status.
- **Error**: Returns an error message if the deletion is unauthorized.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/subsidies/subsidy_id?email=provider@example.com
```

### **Example Response**
```json
{
  "type": "crop",
  "locations": ["Haryana", "Punjab"],
  "max_recipients": 100,
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "submitted_by": "provider_id",
  "status": "withdrawn",
  "$id": "subsidy_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **7. Create a Subsidy Request**

### **Endpoint**
`POST /subsidy_requests`

### **Description**
Creates a request for a subsidy. Only farmers can create requests for approved subsidies.

### **Who Can Call It**
Farmers.

### **Request Body**
```json
{
  "subsidy_id": "string"
}
```

### **Query Parameters**
- `email` (string): The email of the farmer creating the request.

### **Response**
- **Success**: Returns the created request document.
- **Error**: Returns an error message if the subsidy is not approved or if the farmer is not eligible.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/subsidy_requests?email=farmer@example.com \
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
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

# API Documentation for Remaining `/subsidy_requests` Endpoints

This section provides details about the remaining endpoints for managing subsidy requests.

---

## **8. Get Subsidy Requests**

### **Endpoint**
`GET /subsidy_requests`

### **Description**
Fetches subsidy requests based on the user's role and optional filters.

### **Who Can Call It**
- Farmers: Fetch requests they created.
- Providers: Fetch requests for subsidies they created.
- Admins: Fetch all requests.

### **Query Parameters**
- `email` (string): The email of the user fetching requests.
- `status` (string, optional): Filter by request status (`requested`, `accepted`, `rejected`, `withdrawn`, `removed`,`fulfilled` or `all`).
- `subsidy_id` (string, optional): Filter by subsidy ID.

### **Response**
- **Success**: Returns a list of subsidy requests.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/subsidy_requests?email=farmer@example.com&status=requested"
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
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **9. Accept a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/accept`

### **Description**
Accepts a subsidy request. Only admins or the provider who created the subsidy can accept requests.

### **Who Can Call It**
- Admins.
- Providers (only for subsidies they created).

### **Path Parameters**
- `request_id` (string): The ID of the request to accept.

### **Query Parameters**
- `email` (string): The email of the user accepting the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `accepted`.
- **Error**: Returns an error message if the request is not in a `requested` state or if the user is unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/accept?email=admin@example.com
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "accepted",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **10. Reject a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/reject`

### **Description**
Rejects a subsidy request. Only admins or the provider who created the subsidy can reject requests.

### **Who Can Call It**
- Admins.
- Providers (only for subsidies they created).

### **Path Parameters**
- `request_id` (string): The ID of the request to reject.

### **Query Parameters**
- `email` (string): The email of the user rejecting the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `rejected`.
- **Error**: Returns an error message if the request is not in a `requested` state or if the user is unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/reject?email=provider@example.com
```

### **Example Response**
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "rejected",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **11. Delete a Subsidy Request**

### **Endpoint**
`DELETE /subsidy_requests/{request_id}`

### **Description**
Deletes a subsidy request. Farmers can delete their own requests if they are in the `requested` state. Admins can delete if they are in accepted or requested state.

### **Who Can Call It**
- Farmers (only for their own requests in the `requested` state).
- Admins.

### **Path Parameters**
- `request_id` (string): The ID of the request to delete.

### **Query Parameters**
- `email` (string): The email of the user deleting the request.

### **Response**
- **Success**: Returns a success message.
- **Error**: Returns an error message if the request is not in a deletable state or if the user is unauthorized.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/subsidy_requests/request_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "message": "Request deleted successfully"
}
```


---

## **12. Fulfill a Subsidy Request**

### **Endpoint**
`PATCH /subsidy_requests/{request_id}/fulfill`

### **Description**
Marks a subsidy request as fulfilled. Only the farmer who created the request or an admin can fulfill it. The request must be in the `accepted` state to be fulfilled.

### **Who Can Call It**
- Farmers (only for their own requests).
- Admins.

### **Path Parameters**
- `request_id` (string): The ID of the request to fulfill.

### **Query Parameters**
- `email` (string): The email of the user fulfilling the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `fulfilled`.
- **Error**: Returns an error message if the request is not in a fulfillable state or if the user is unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/subsidy_requests/request_id/fulfill?email=farmer@example.com
```

### **Example Response**
**Success**:
```json
{
  "subsidy_id": "subsidy_id",
  "farmer_id": "farmer_id",
  "status": "fulfilled",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

**Error (Unauthorized)**:
```json
{
  "detail": "Only the farmer who created the request or an admin can fulfill requests"
}
```

**Error (Invalid State)**:
```json
{
  "detail": "Request not available for fulfillment"
}
```