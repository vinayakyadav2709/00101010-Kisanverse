note: zipcode have been added. document api response may not show them. 
# API Documentation for `/users` Endpoint

This document provides details about the API endpoints defined in the users.py file. It includes information about who can call the endpoints, required parameters, request bodies, and example API calls with expected outputs.

---

## **1. Register a User**

### **Endpoint**
`POST /users`

### **Description**
Registers a new user in the system.

### **Who Can Call It**
Anyone.

### **Request Body**
```json
{
  "name": "string",
  "email": "string",
  "role": "string (admin, farmer, buyer, provider)",
  "address": "string",
  "zipcode": "string"
}
```

### **Response**
- **Success**: Returns the created user document.
- **Error**: Returns an error message if the user already exists or the role is invalid.

### **Example API Call**
```bash
curl -X POST http://localhost:8000/users \
-H "Content-Type: application/json" \
-d '{
  "name": "John Doe",
  "email": "john.doe@example.com",
  "role": "admin",
  "address": "123 Main St"
  "zipcode" : "12345"
}'
```

### **Example Response**
```json
{
  "name": "John Doe",
  "role": "admin",
  "address": "123 Main St",
  "email": "john.doe@example.com",
  "$id": "680cd1cb0021422a786e",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00",
  "$permissions": [],
  "$databaseId": "agri_marketplace",
  "$collectionId": "users"
}
```

---

## **2. Get a User by Email**

### **Endpoint**
`GET /users/{email}`

### **Description**
Fetches a user by their email.

### **Who Can Call It**
Anyone.

### **Path Parameters**
- `email` (string): The email of the user.

### **Response**
- **Success**: Returns the user document.
- **Error**: Returns an error message if the user does not exist.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/users/john.doe@example.com
```

### **Example Response**
```json
{
  "name": "John Doe",
  "role": "admin",
  "address": "123 Main St",
  "email": "john.doe@example.com",
  "$id": "680cd1cb0021422a786e",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:30:03.816+00:00",
  "$permissions": [],
  "$databaseId": "agri_marketplace",
  "$collectionId": "users"
}
```

---

## **3. List Users**

### **Endpoint**
`GET /users`

### **Description**
Lists all users or filters users by role.

### **Who Can Call It**
Anyone.

### **Query Parameters**
- `type` (string, optional): Filter by role (`admin`, `farmer`, `buyer`, `provider`, or `all`).

### **Response**
- **Success**: Returns a list of user documents.
- **Error**: Returns an error message if the type is invalid.

### **Example API Call**
```bash
curl -X GET "http://localhost:8000/users?type=all"
```

### **Example Response**
```json
{
  "total": 5,
  "documents": [
    {
      "name": "Admin User",
      "role": "admin",
      "address": "Admin Address",
      "email": "admin@example.com",
      "$id": "680ba2e30015e27516ac",
      "$createdAt": "2025-04-25T14:57:39.424+00:00",
      "$updatedAt": "2025-04-25T14:57:39.424+00:00",
      "$permissions": [],
      "$databaseId": "agri_marketplace",
      "$collectionId": "users"
    },
    ...
  ]
}
```

---

## **4. Update a User**

### **Endpoint**
`PATCH /users/{email}`

### **Description**
Updates a user's details. Only the user themselves or an admin can update a user. only admin can update role, except themselves. 

### **Who Can Call It**
- The user themselves.
- Admins.

### **Path Parameters**
- `email` (string): The email of the user to update.

### **Request Body**
```json
{
  "name": "string (optional)",
  "email": "string (optional)",
  "role": "string (optional, admin only)",
  "address": "string (optional)",
  "zipcode": "int (optional)"
}
```

### **Response**
- **Success**: Returns the updated user document.
- **Error**: Returns an error message if the update is invalid or unauthorized.

### **Example API Call**
```bash
curl -X PATCH http://localhost:8000/users/john.doe@example.com \
-H "Content-Type: application/json" \
-d '{
  "address": "456 Updated St"
}'
```

### **Example Response**
```json
{
  "name": "John Doe",
  "role": "admin",
  "address": "456 Updated St",
  "email": "john.doe@example.com",
  "$id": "680cd1cb0021422a786e",
  "$createdAt": "2025-04-26T12:30:03.816+00:00",
  "$updatedAt": "2025-04-26T12:45:03.816+00:00",
  "$permissions": [],
  "$databaseId": "agri_marketplace",
  "$collectionId": "users"
}
```

---

## **5. Delete a User**

### **Endpoint**
`DELETE /users/{email}`

### **Description**
Deletes a user by their email. Only the user themselves or an admin can delete a user.

### **Who Can Call It**
- The user themselves.
- Admins.

### **Path Parameters**
- `email` (string): The email of the user to delete.

### **Query Parameters**
- `requester_email` (string): The email of the requester.

### **Response**
- **Success**: Returns a success message.
- **Error**: Returns an error message if the deletion is unauthorized.

### **Example API Call**
```bash
curl -X DELETE "http://localhost:8000/users/john.doe@example.com?requester_email=admin@example.com"
```

### **Example Response**
```json
{
  "message": "User deleted successfully"
}
```

---

## **6. Get User Role**

### **Endpoint**
`GET /users/{email}/role`

### **Description**
Fetches the role of a user by their email.

### **Who Can Call It**
Anyone.

### **Path Parameters**
- `email` (string): The email of the user.

### **Response**
- **Success**: Returns the user's role.
- **Error**: Returns an error message if the user does not exist.

### **Example API Call**
```bash
curl -X GET http://localhost:8000/users/john.doe@example.com/role
```

### **Example Response**
```json
{
  "role": "admin"
}
```