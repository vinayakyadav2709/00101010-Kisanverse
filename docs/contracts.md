# API Documentation for `/contracts` and `/contract_requests` Endpoints

This document provides details about the API endpoints defined in the contracts.py file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Get Contracts**

### **Endpoint**
`GET /contracts`

### **Description**
Fetches contracts based on optional filters.

### **Who Can Call It**
- Buyers: Fetch contracts they created.
- Farmers: Fetch contracts applicable to their location.
- Admins: Fetch all contracts.

### **Query Parameters**
- `email` (string, optional): The email of the user fetching contracts.
- `template_type` (string, optional): Filter by contract type (`seed`, `machine`, `fert`, `general`, or `all`).

### **Response**
- **Success**: Returns a list of contracts.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/contracts?email=buyer@example.com&template_type=seed"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "template_type": "seed",
      "locations": ["Haryana", "Punjab"],
      "dynamic_fields": "{\"field1\": \"value1\"}",
      "buyer_id": "buyer_id",
      "status": "listed",
      "$id": "contract_id_1",
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **2. Create a Contract**

### **Endpoint**
`POST /contracts`

### **Description**
Creates a new contract. Only buyers can create contracts.

### **Who Can Call It**
Buyers.

### **Request Body**
```json
{
  "template_type": "string (seed, machine, fert, general)",
  "locations": ["string"],
  "dynamic_fields": "string (valid JSON)"
}
```

### **Query Parameters**
- `email` (string): The email of the buyer creating the contract.

### **Response**
- **Success**: Returns the created contract document.
- **Error**: Returns an error message if the user is not a buyer or if the input is invalid.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/contracts?email=buyer@example.com \
-H "Content-Type: application/json" \
-d '{
  "template_type": "seed",
  "locations": ["Haryana", "Punjab"],
  "dynamic_fields": "{\"field1\": \"value1\"}"
}'
```

### **Example Response**
```json
{
  "template_type": "seed",
  "locations": ["Haryana", "Punjab"],
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "buyer_id": "buyer_id",
  "status": "listed",
  "$id": "contract_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **3. Update a Contract**

### **Endpoint**
`PATCH /contracts/{contract_id}`

### **Description**
Updates a contract. Only admins can update contracts.

### **Who Can Call It**
Admins.

### **Path Parameters**
- `contract_id` (string): The ID of the contract to update.

### **Request Body**
```json
{
  "template_type": "string (optional)",
  "locations": ["string (optional)"],
  "dynamic_fields": "string (optional, valid JSON)",
  "status": "string (optional, only 'fulfilled' allowed)"
}
```

### **Response**
- **Success**: Returns the updated contract document.
- **Error**: Returns an error message if the update is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/contracts/contract_id?email=admin@example.com \
-H "Content-Type: application/json" \
-d '{
  "status": "fulfilled"
}'
```

### **Example Response**
```json
{
  "template_type": "seed",
  "locations": ["Haryana", "Punjab"],
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "buyer_id": "buyer_id",
  "status": "fulfilled",
  "$id": "contract_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **4. Delete a Contract**

### **Endpoint**
`DELETE /contracts/{contract_id}`

### **Description**
Deletes a contract. Buyers can cancel their own contracts, while admins can remove any contract.

### **Who Can Call It**
- Buyers (only for their own contracts).
- Admins.

### **Path Parameters**
- `contract_id` (string): The ID of the contract to delete.

### **Query Parameters**
- `email` (string): The email of the user deleting the contract.

### **Response**
- **Success**: Returns the updated contract document with the status set to `cancelled` or `removed`.
- **Error**: Returns an error message if the deletion is unauthorized.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/contracts/contract_id?email=buyer@example.com
```

### **Example Response**
```json
{
  "template_type": "seed",
  "locations": ["Haryana", "Punjab"],
  "dynamic_fields": "{\"field1\": \"value1\"}",
  "buyer_id": "buyer_id",
  "status": "cancelled",
  "$id": "contract_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **5. Get Contract Requests**

### **Endpoint**
`GET /contract_requests`

### **Description**
Fetches contract requests based on optional filters.

### **Who Can Call It**
- Farmers: Fetch requests they created.
- Buyers: Fetch requests for their contracts.
- Admins: Fetch all requests.

### **Query Parameters**
- `email` (string, optional): The email of the user fetching requests.
- `status` (string, optional): Filter by request status (`pending`, `accepted`, `rejected`, `withdrawn`, `fulfilled`, `removed`, or `all`).
- `contract_id` (string, optional): Filter by contract ID.

### **Response**
- **Success**: Returns a list of contract requests.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/contract_requests?email=farmer@example.com&status=pending"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "contract_id": "contract_id",
      "farmer_id": "farmer_id",
      "status": "pending",
      "$id": "request_id_1",
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **6. Create a Contract Request**

### **Endpoint**
`POST /contract_requests`

### **Description**
Creates a request for a contract. Only farmers can create requests for listed contracts.

### **Who Can Call It**
Farmers.

### **Request Body**
```json
{
  "contract_id": "string"
}
```

### **Query Parameters**
- `email` (string): The email of the farmer creating the request.

### **Response**
- **Success**: Returns the created request document.
- **Error**: Returns an error message if the contract is not listed or if the farmer is not eligible.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/contract_requests?email=farmer@example.com \
-H "Content-Type: application/json" \
-d '{
  "contract_id": "contract_id"
}'
```

### **Example Response**
```json
{
  "contract_id": "contract_id",
  "farmer_id": "farmer_id",
  "status": "pending",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **7. Accept a Contract Request**

### **Endpoint**
`PATCH /contract_requests/{request_id}/accept`

### **Description**
Accepts a contract request. Only the buyer who created the contract or an admin can accept requests.

### **Who Can Call It**
- Buyers (only for their own contracts).
- Admins.

### **Path Parameters**
- `request_id` (string): The ID of the request to accept.

### **Query Parameters**
- `email` (string): The email of the user accepting the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `accepted`.
- **Error**: Returns an error message if the request is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/contract_requests/request_id/accept?email=buyer@example.com
```

### **Example Response**
```json
{
  "contract_id": "contract_id",
  "farmer_id": "farmer_id",
  "status": "accepted",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

# API Documentation for Remaining `/contract_requests` Endpoints

This section provides details about the remaining endpoints for managing contract requests.

---

## **8. Reject a Contract Request**

### **Endpoint**
`PATCH /contract_requests/{request_id}/reject`

### **Description**
Rejects a contract request. Only the buyer who created the contract or an admin can reject requests.

### **Who Can Call It**
- Buyers (only for their own contracts).
- Admins.

### **Path Parameters**
- `request_id` (string): The ID of the request to reject.

### **Query Parameters**
- `email` (string): The email of the user rejecting the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `rejected`.
- **Error**: Returns an error message if the request is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/contract_requests/request_id/reject?email=buyer@example.com
```

### **Example Response**
```json
{
  "contract_id": "contract_id",
  "farmer_id": "farmer_id",
  "status": "rejected",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **9. Delete a Contract Request**

### **Endpoint**
`DELETE /contract_requests/{request_id}`

### **Description**
Deletes a contract request. Farmers can delete their own requests if they are in the `pending` state. Admins can delete any request.

### **Who Can Call It**
- Farmers (only for their own requests in the `pending` state).
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
curl -X DELETE http://localhost:8000/contract_requests/request_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "message": "Request deleted successfully"
}
```

---

## **10. Fulfill a Contract Request**

### **Endpoint**
`PATCH /contract_requests/{request_id}/fulfill`

### **Description**
Marks a contract request as fulfilled. Only the buyer who created the contract or an admin can mark it as fulfilled.

### **Who Can Call It**
- Buyers (only for their own contracts).
- Admins.

### **Path Parameters**
- `request_id` (string): The ID of the request to fulfill.

### **Query Parameters**
- `email` (string): The email of the user fulfilling the request.

### **Response**
- **Success**: Returns the updated request document with the status set to `fulfilled`.
- **Error**: Returns an error message if the request is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/contract_requests/request_id/fulfill?email=buyer@example.com
```

### **Example Response**
```json
{
  "contract_id": "contract_id",
  "farmer_id": "farmer_id",
  "status": "fulfilled",
  "$id": "request_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **11. Reject All Pending Requests for a Contract**

### **Function**
`reject_pending_requests(contract_id: str)`

### **Description**
Automatically rejects all pending requests for a contract when the contract is marked as `fulfilled` or `removed`.

### **Who Can Call It**
This function is called internally when a contract is marked as `fulfilled` or `removed`.

### **Parameters**
- `contract_id` (string): The ID of the contract for which pending requests should be rejected.

### **Response**
- **Success**: Returns a success message.
- **Error**: Raises an HTTPException if there is an error rejecting requests.

### **Example Usage**
This function is not exposed as an API endpoint but is triggered internally.

---
