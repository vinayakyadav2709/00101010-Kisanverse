from fastapi import APIRouter, HTTPException
from appwrite.id import ID
from core.config import DATABASE_ID, DATABASES, COLLECTION_USERS
from core.dependencies import (
    get_user_by_email,
    get_user_by_email_or_raise,
    get_document_or_raise,
)
from typing import List, Optional
from pydantic import BaseModel, field_validator
import time
from appwrite.query import Query

router = APIRouter(prefix="/users", tags=["Users"])


class RegisterUserModel(BaseModel):
    name: str
    email: str
    role: str
    address: str
    zipcode: str


class UpdateUserModel(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None
    address: Optional[str] = None
    zipcode: Optional[str] = None


@router.post("")
def register_user(payload: RegisterUserModel):
    if get_user_by_email(payload.email):
        raise HTTPException(status_code=400, detail="User already exists")
    if payload.role not in ["admin", "farmer", "buyer"]:
        raise HTTPException(status_code=400, detail="Invalid role")
    data = payload.model_dump()

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            unique_id = ID.unique()
            return DATABASES.create_document(
                DATABASE_ID, COLLECTION_USERS, unique_id, data
            )

        except Exception as e:
            if "already exists" in str(e) and attempt < max_attempts - 1:
                time.sleep(0.1)  # Optional: short delay before retry
                continue
            raise HTTPException(status_code=400, detail=str(e))


@router.get("/{email}")
def get_user(email: str):  # -> dict:
    return get_user_by_email_or_raise(email)


@router.get("")
def list_users(type: Optional[str] = "all"):
    try:
        # Prepare filters
        query_filters = []

        # If type is provided, filter users based on it
        if type == "admin":
            query_filters.append(Query.equal("role", ["admin"]))
        elif type == "farmer":
            query_filters.append(Query.equal("role", ["farmer"]))
        elif type == "buyer":
            query_filters.append(Query.equal("role", ["buyer"]))
        elif type == "all":
            # No additional filters for "all"
            pass
        else:
            raise HTTPException(status_code=400, detail="Invalid type provided")
        query_filters.append(Query.limit(10000))  # Limit to 100 results
        # List documents based on filters
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_USERS, queries=query_filters
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching users: {str(e)}")


@router.patch("/{email}")
def update_user(
    payload: UpdateUserModel, email: str, requester_email: Optional[str] = None
):
    user = get_user_by_email_or_raise(email)
    requestor_email = requester_email or email
    requester = get_user_by_email_or_raise(requestor_email)

    user_role = user["role"]
    user_id = user["$id"]
    if requester["$id"] != user_id and requester["role"] != "admin":
        raise HTTPException(
            status_code=403,
            detail="Permission denied, only admin and user itself can update user",
        )

    updates = {}
    if payload.name:
        updates["name"] = payload.name
    if payload.email:
        updates["email"] = payload.email
    if payload.role:
        if payload.role != user_role:
            if requester["role"] == "admin" and requester["$id"] != user_id:
                updates["role"] = payload.role
            elif requester["$id"] == user_id and requester["role"] == "admin":
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied, admin cannot change their own role",
                )
            else:
                raise HTTPException(
                    status_code=403,
                    detail="Permission denied, only admin can change roles",
                )
    if payload.address:
        updates["address"] = payload.address
    if payload.zipcode:
        updates["zipcode"] = payload.zipcode

    if not updates:
        raise HTTPException(status_code=400, detail="No valid updates provided")

    try:
        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_USERS, user_id, updates
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{email}")
def delete_user_by_email(email: str, requester_email: str):
    user_id = get_user_by_email_or_raise(email)["$id"]
    requester = get_user_by_email_or_raise(requester_email)
    if requester["$id"] != user_id and requester["role"] != "admin":
        raise HTTPException(status_code=403, detail="Permission denied")
    try:
        DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, user_id)
        return {"message": "User deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{email}/role")
def get_user_type(email: str):
    return {"role": get_user_by_email_or_raise(email)["role"]}


def create_all_user_types():
    roles = ["admin", "farmer", "buyer"]
    for role in roles:
        payload = RegisterUserModel(
            name=f"{role.capitalize()} User",
            email=f"{role}@example.com",
            role=role,
            address=f"{role.capitalize()} Address",
        )
        try:
            print(f"Registering {role} user...")
            response = register_user(payload)
            print(response)
        except HTTPException as e:
            print(f"Error registering {role} user: {e.detail}")


def run_tests():
    # Test case: Register a new user
    print("Test: Register a new user")
    payload = RegisterUserModel(
        name="John Doe",
        email="john.doe@example.com",
        role="admin",
        address="123 Main St",
    )
    try:
        print(register_user(payload))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Try to register a duplicate user
    print("\nTest: Register a duplicate user")
    try:
        print(register_user(payload))
        print(
            register_user(
                RegisterUserModel(
                    name="John Doe",
                    email="john.doe@example.com",
                    role="farmer",
                    address="123 Main St",
                )
            )
        )
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get an existing user
    print("\nTest: Get an existing user")
    try:
        print(get_user("john.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get a user that doesn't exist
    print("\nTest: Get a user that doesn't exist")
    try:
        print(get_user("nonexistent@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: List all users
    print("\nTest: List all users")
    try:
        print(list_users("all"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: List users by role (e.g., farmers)
    print("\nTest: List users by role (farmers)")
    try:
        print(list_users("farmer"))
        print(list_users("admin"))
        print(list_users("buyer"))
        print(list_users("all"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update self (valid update)
    print("\nTest: Update self (valid update)")
    payload = UpdateUserModel(
        email="john.doe@example.com",
        address="456 Updated St",
    )
    try:
        print(update_user(payload, "john.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Admin updates another user
    print("\nTest: Admin updates another user")
    payload = RegisterUserModel(
        name="Jane Doe",
        email="jane.doe@example.com",
        role="buyer",
        address="789 Main St",
    )
    update_payload = UpdateUserModel(
        name="Jane Updated",
        email="jane.doe@example.com",
        address="789 Updated St",
    )
    try:
        print(register_user(payload))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(
            update_user(update_payload, "john.doe@example.com", "jane.doe@example.com")
        )
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Admin tries to update another user's role
    print("\nTest: Admin tries to update another user's role")
    update_payload = UpdateUserModel(
        role="buyer",
        email="jane.doe@example.com",
    )

    try:
        print(
            update_user(update_payload, "jane.doe@example.com", "john.doe@example.com")
        )
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: User tries to update their own role
    print("\nTest: User tries to update their own role")
    update_payload = UpdateUserModel(
        role="buyer",
        email="jane.doe@example.com",
        requester_email="jane.doe@example.com",
    )
    try:
        print(update_user(update_payload, "jane.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Admin tries to update their own role
    print("\nTest: Admin tries to update their own role")
    update_payload = UpdateUserModel(
        role="buyer",
        email="john.doe@example.com",
    )
    try:
        print(
            update_user(update_payload, "john.doe@example.com", "john.doe@example.com")
        )
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Non-admin tries to delete another user
    print("\nTest: Non-admin tries to delete another user")
    try:
        print(delete_user_by_email("john.doe@example.com", "jane.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Admin deletes another user
    print("\nTest: Admin deletes another user")

    try:
        print(delete_user_by_email("jane.doe@example.com", "john.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a user that doesn't exist
    print("\nTest: Delete a user that doesn't exist")
    try:
        print(delete_user_by_email("nonexistent@example.com", "john.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete self
    print("\nTest: Delete self")
    try:
        print(delete_user_by_email("john.doe@example.com", "john.doe@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
