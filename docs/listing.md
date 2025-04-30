# API Documentation for `/listing` and `/bids` Endpoints

This document provides details about the API endpoints defined in the listings.py file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Get Crop Listings**

### **Endpoint**
`GET /listing`

### **Description**
Fetches crop listings based on optional filters.

### **Who Can Call It**
- Farmers: Fetch their own listings.
- Admins: Fetch all listings.

### **Query Parameters**
- `email` (string, optional): The email of the farmer (for admin filtering).
- `type` (string, optional): Filter by listing status (`listed`, `removed`, `cancelled`, `fulfilled`, or `all`).

### **Response**
- **Success**: Returns a list of crop listings.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/listing?email=farmer@example.com&type=listed"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "crop_type": "Wheat",
      "price_per_kg": 20.0,
      "total_quantity": 100.0,
      "available_quantity": 80.0,
      "status": "listed",
      "farmer_id": "farmer_id",
      "$id": "listing_id_1",
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **2. Get a Specific Crop Listing**

### **Endpoint**
`GET /listing/{listing_id}`

### **Description**
Fetches a specific crop listing by its ID.

### **Who Can Call It**
Anyone.

### **Path Parameters**
- `listing_id` (string): The ID of the crop listing.

### **Response**
- **Success**: Returns the crop listing document.
- **Error**: Returns an error message if the listing does not exist.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/listing/listing_id
```

### **Example Response**
```json
{
  "crop_type": "Wheat",
  "price_per_kg": 20.0,
  "total_quantity": 100.0,
  "available_quantity": 80.0,
  "status": "listed",
  "farmer_id": "farmer_id",
  "$id": "listing_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **3. Create a Crop Listing**

### **Endpoint**
`POST /listing`

### **Description**
Creates a new crop listing. Only farmers can create listings.

### **Who Can Call It**
Farmers.

### **Request Body**
```json
{
  "crop_type": "string",
  "price_per_kg": "float",
  "total_quantity": "float"
}
```

### **Query Parameters**
- `email` (string): The email of the farmer creating the listing.

### **Response**
- **Success**: Returns the created crop listing document.
- **Error**: Returns an error message if the user is not a farmer or if the input is invalid.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/listing?email=farmer@example.com \
-H "Content-Type: application/json" \
-d '{
  "crop_type": "Wheat",
  "price_per_kg": 20.0,
  "total_quantity": 100.0
}'
```

### **Example Response**
```json
{
  "crop_type": "Wheat",
  "price_per_kg": 20.0,
  "total_quantity": 100.0,
  "available_quantity": 100.0,
  "status": "listed",
  "farmer_id": "farmer_id",
  "$id": "listing_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **4. Update a Crop Listing**

### **Endpoint**
`PATCH /listing/{listing_id}`

### **Description**
Updates a crop listing. Only the farmer who created the listing or an admin can update it and only if its listed.

### **Who Can Call It**
- Farmers (only for their own listings).
- Admins.

### **Path Parameters**
- `listing_id` (string): The ID of the crop listing to update.

### **Request Body**
```json
{
  "crop_type": "string (optional)",
  "price_per_kg": "float (optional)",
  "total_quantity": "float (optional)"
}
```

### **Response**
- **Success**: Returns the updated crop listing document.
- **Error**: Returns an error message if the update is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/listing/listing_id?email=farmer@example.com \
-H "Content-Type: application/json" \
-d '{
  "price_per_kg": 25.0,
  "total_quantity": 120.0
}'
```

### **Example Response**
```json
{
  "crop_type": "Wheat",
  "price_per_kg": 25.0,
  "total_quantity": 120.0,
  "available_quantity": 100.0,
  "status": "listed",
  "farmer_id": "farmer_id",
  "$id": "listing_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **5. Cancel a Crop Listing**

### **Endpoint**
`DELETE /listing/{listing_id}`

### **Description**
Cancels a crop listing. Farmers can cancel their own listings, while admins can remove any listing. can only be if listing is in listed state. 

### **Who Can Call It**
- Farmers (only for their own listings).
- Admins.

### **Path Parameters**
- `listing_id` (string): The ID of the crop listing to cancel.

### **Query Parameters**
- `email` (string): The email of the user canceling the listing.

### **Response**
- **Success**: Returns the updated crop listing document with the status set to `cancelled` or `removed`.
- **Error**: Returns an error message if the cancellation is unauthorized.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/listing/listing_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "crop_type": "Wheat",
  "price_per_kg": 20.0,
  "total_quantity": 100.0,
  "available_quantity": 80.0,
  "status": "cancelled",
  "farmer_id": "farmer_id",
  "$id": "listing_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **6. Place a Bid**

### **Endpoint**
`POST /bids`

### **Description**
Places a bid on a crop listing. Only buyers can place bids.

### **Who Can Call It**
Buyers.

### **Request Body**
```json
{
  "quantity": "float (optional)",
  "price_per_kg": "float (optional)",
  "listing_id": "string"
}
```

### **Query Parameters**
- `email` (string): The email of the buyer placing the bid.

### **Response**
- **Success**: Returns the created bid document.
- **Error**: Returns an error message if the bid is invalid or unauthorized.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/bids?email=buyer@example.com \
-H "Content-Type: application/json" \
-d '{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id"
}'
```

### **Example Response**
```json
{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "pending",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---

## **7. Accept a Bid**

### **Endpoint**
`PATCH /bids/{bid_id}/accept`

### **Description**
Accepts a bid. Only the farmer who created the listing can accept bids.

### **Who Can Call It**
Farmers (only for their own listings).

### **Path Parameters**
- `bid_id` (string): The ID of the bid to accept.

### **Query Parameters**
- `email` (string): The email of the farmer accepting the bid.

### **Response**
- **Success**: Returns the updated bid document with the status set to `accepted`.
- **Error**: Returns an error message if the bid is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/bids/accept/bid_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "accepted",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

# API Documentation for Remaining `/bids` Endpoints

This section provides details about the remaining endpoints for managing bids.

---

## **8. Reject a Bid**

### **Endpoint**
`PATCH /bids/{bid_id}/reject`

### **Description**
Rejects a bid. Only the farmer who created the listing can reject bids.

### **Who Can Call It**
Farmers (only for their own listings).

### **Path Parameters**
- `bid_id` (string): The ID of the bid to reject.

### **Query Parameters**
- `email` (string): The email of the farmer rejecting the bid.

### **Response**
- **Success**: Returns the updated bid document with the status set to `rejected`.
- **Error**: Returns an error message if the bid is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/bids/reject/bid_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "rejected",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **9. Update a Bid**

### **Endpoint**
`PATCH /bids/{bid_id}`

### **Description**
Updates a bid. Only the buyer who placed the bid and admin can update it. 
admin can only update pending and accepted bids and buyer can only update pending bid.

### **Who Can Call It**
Buyers (only for their own bids).

### **Path Parameters**
- `bid_id` (string): The ID of the bid to update.

### **Query Parameters**
- `email` (string): The email of the buyer updating the bid.

### **Request Body**
```json
{
  "quantity": "float (optional)",
  "price_per_kg": "float (optional)",
  "listing_id": "string (optional)"
}
```

### **Response**
- **Success**: Returns the updated bid document.
- **Error**: Returns an error message if the update is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/bids/update/bid_id?email=buyer@example.com \
-H "Content-Type: application/json" \
-d '{
  "quantity": 15.0,
  "price_per_kg": 22.0
}'
```

### **Example Response**
```json
{
  "quantity": 15.0,
  "price_per_kg": 22.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "pending",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **10. Delete a Bid**

### **Endpoint**
`DELETE /bids/{bid_id}`

### **Description**
Deletes a bid. Only the buyer who placed the bid or an admin can delete it.
buyer can only delete pending bids and admin can only delete accepted or pending bids. 

### **Who Can Call It**
- Buyers (only for their own bids).
- Admins.

### **Path Parameters**
- `bid_id` (string): The ID of the bid to delete.

### **Query Parameters**
- `email` (string): The email of the user deleting the bid.

### **Response**
- **Success**: Returns a success message.
- **Error**: Returns an error message if the deletion is unauthorized.

### **Example API Call**
```bash
curl -X DELETE http://localhost:8000/bids/bid_id?email=buyer@example.com
```

### **Example Response**
```json
{
  "message": "Bid deleted successfully"
}
```

---

## **11. Fulfill a Bid**

### **Endpoint**
`PATCH /bids/{bid_id}/fulfill`

### **Description**
Marks a bid as fulfilled. Only the farmer who created the listing or the buyer who placed the bid can mark it as fulfilled.

### **Who Can Call It**
- Farmers (only for their own listings).
- Buyers (only for their own bids).

### **Path Parameters**
- `bid_id` (string): The ID of the bid to fulfill.

### **Query Parameters**
- `email` (string): The email of the user fulfilling the bid.

### **Response**
- **Success**: Returns the updated bid document with the status set to `fulfilled`.
- **Error**: Returns an error message if the bid is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/bids/fulfill/bid_id?email=farmer@example.com
```

### **Example Response**
```json
{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "fulfilled",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00"
}
```

---

## **12. Get All Bids**

### **Endpoint**
`GET /bids`

### **Description**
Fetches bids based on optional filters.

### **Who Can Call It**
- Farmers: Fetch bids on their own listings.
- Buyers: Fetch their own bids.
- Admins: Fetch all bids.

### **Query Parameters**
- `email` (string, optional): The email of the user fetching bids.
- `type` (string, optional): Filter by bid status (`pending`, `accepted`, `rejected`, `fulfilled`, or `all`).
- `listing_id` (string, optional): Filter by listing ID.

### **Response**
- **Success**: Returns a list of bids.
- **Error**: Returns an error message if the filters are invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/bids?email=farmer@example.com&type=pending&listing_id=listing_id"
```

### **Example Response**
```json
{
  "total": 2,
  "documents": [
    {
      "quantity": 10.0,
      "price_per_kg": 20.0,
      "listing_id": "listing_id",
      "buyer_id": "buyer_id",
      "status": "pending",
      "$id": "bid_id_1",
      "$createdAt": "2025-04-26T12:30:03.816+00:00",
      "$updatedAt": "2025-04-26T12:30:03.816+00:00"
    },
    ...
  ]
}
```

---

## **13. Get a Specific Bid**

### **Endpoint**
`GET /bids/{bid_id}`

### **Description**
Fetches a specific bid by its ID.

### **Who Can Call It**
Anyone.

### **Path Parameters**
- `bid_id` (string): The ID of the bid.

### **Response**
- **Success**: Returns the bid document.
- **Error**: Returns an error message if the bid does not exist.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/bids/bid_id
```

### **Example Response**
```json
{
  "quantity": 10.0,
  "price_per_kg": 20.0,
  "listing_id": "listing_id",
  "buyer_id": "buyer_id",
  "status": "pending",
  "$id": "bid_id",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00"
}
```

---
