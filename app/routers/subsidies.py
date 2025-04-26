from fastapi import APIRouter, HTTPException

from core.config import (
    DATABASE_ID,
    DATABASES,
    COLLECTION_SUBSIDIES,
    COLLECTION_SUBSIDY_REQUESTS,
    COLLECTION_USERS,
)
from core.dependencies import get_document_or_raise, get_user_by_email_or_raise
from pydantic import BaseModel, field_validator
from typing import List, Optional
from appwrite.query import Query
import json
from appwrite.id import ID

subsidies_router = APIRouter(prefix="/subsidies", tags=["Subsidies"])

# Enums for type and status
SUBSIDY_TYPES = ["crop", "seed", "fertilizer", "machine", "general"]
SUBSIDY_STATUSES = [
    "pending",
    "approved",
    "rejected",
    "removed",
    "fulfilled",
    "withdrawn",
]


# Pydantic models
class SubsidyCreateModel(BaseModel):
    type: str
    locations: List[str]
    max_recipients: int
    dynamic_fields: str

    @field_validator("type")
    @classmethod
    def validate_type(cls, value):
        if value not in SUBSIDY_TYPES:
            raise ValueError(f"Invalid type. Must be one of {SUBSIDY_TYPES}")
        return value

    @field_validator("dynamic_fields")
    @classmethod
    def validate_dynamic_fields(cls, value):
        try:
            json.loads(value)  # Ensure it's valid JSON
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for dynamic_fields")
        return value


class SubsidyUpdateModel(BaseModel):
    type: Optional[str] = None
    locations: Optional[List[str]] = None
    max_recipients: Optional[int] = None
    dynamic_fields: Optional[str] = None
    status: Optional[str] = None

    @field_validator("type", mode="before")
    @classmethod
    def validate_type(cls, value):
        if value and value not in SUBSIDY_TYPES:
            raise ValueError(f"Invalid type. Must be one of {SUBSIDY_TYPES}")
        return value

    @field_validator("dynamic_fields", mode="before")
    @classmethod
    def validate_dynamic_fields(cls, value):
        if value:
            try:
                json.loads(value)  # Ensure it's valid JSON
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for dynamic_fields")
        return value


@subsidies_router.post("")
def create_subsidy(data: SubsidyCreateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        role = user["role"]
        if role != "provider":
            raise HTTPException(
                status_code=403, detail="Permission denied. only for provider"
            )
        subsidy_data = data.model_dump()
        subsidy_data["submitted_by"] = user["$id"]
        subsidy_data["status"] = "pending"

        # Retry mechanism for ID uniqueness
        for attempt in range(3):
            try:
                return DATABASES.create_document(
                    DATABASE_ID, COLLECTION_SUBSIDIES, ID.unique(), subsidy_data
                )
            except Exception as e:
                if "already exists" in str(e).lower() and attempt < 2:
                    continue  # Retry if ID conflict occurs
                raise HTTPException(
                    status_code=500, detail=f"Error creating subsidy: {str(e)}"
                )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error creating subsidy: {str(e)}")


@subsidies_router.get("")
def get_subsidies(email: Optional[str] = None, type: Optional[str] = None):
    try:
        query_filters = []
        if type and type != "all":
            if type not in SUBSIDY_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {SUBSIDY_STATUSES} or 'all'",
                )
            query_filters.append(Query.equal("status", [type]))
        if email:
            user = get_user_by_email_or_raise(email)
            role = user["role"]

            if role == "provider":
                # Return subsidies created by the provider, optionally filtered by type
                query_filters.append(Query.equal("submitted_by", [user["$id"]]))
                if type:
                    if type not in SUBSIDY_TYPES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid type. Must be one of {SUBSIDY_TYPES}",
                        )
                    query_filters.append(Query.equal("type", [type]))
            elif role == "farmer":
                # Return only approved subsidies applicable to the farmer based on their locations
                farmer_location = user.get(
                    "address", ""
                )  # assuming it's a single string like "Bangalore"
                query_filters.append(Query.contains("locations", [farmer_location]))
                query_filters.append(Query.equal("status", ["approved"]))

            else:
                # Treat other roles (e.g., buyer) like admin
                if type and type != "all":
                    if type not in SUBSIDY_STATUSES:
                        raise HTTPException(
                            status_code=400,
                            detail=f"Invalid status. Must be one of {SUBSIDY_STATUSES} or 'all'",
                        )
                    query_filters.append(Query.equal("status", [type]))
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_SUBSIDIES, queries=query_filters
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching subsidies: {str(e)}"
        )


@subsidies_router.patch("{subsidy_id}")
def update_subsidy(subsidy_id: str, updates: SubsidyUpdateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Permission denied")
        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        updated_data = updates.model_dump(exclude_unset=True)

        # Handle max_recipients updates
        if "max_recipients" in updated_data:
            if subsidy["status"] not in ["approved", "pending"]:
                raise HTTPException(
                    status_code=400,
                    detail="Max recipients can only be updated if the subsidy is approved or pending",
                )
            new_max_recipients = updated_data["max_recipients"]
            current_accepted = subsidy.get("recipients_accepted", 0)

            if new_max_recipients < current_accepted:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"Cannot decrease max_recipients below the current number of "
                        f"accepted recipients ({current_accepted})"
                    ),
                )

            # Update status to fulfilled if max_recipients equals recipients_accepted
            if new_max_recipients == current_accepted:
                updated_data["status"] = "fulfilled"

        # Ensure status can only be changed from pending to approved
        if "status" in updated_data:
            if subsidy["status"] != "pending" or updated_data["status"] != "approved":
                raise HTTPException(
                    status_code=400,
                    detail="Status can only be changed from pending to approved",
                )

        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, updated_data
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error updating subsidy: {str(e)}")


@subsidies_router.patch("{subsidy_id}/reject")
def reject_subsidy(subsidy_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Permission denied")

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        if subsidy["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Only pending subsidies can be rejected"
            )

        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, {"status": "rejected"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rejecting subsidy: {str(e)}"
        )


@subsidies_router.patch("{subsidy_id}/approve")
def approve_subsidy(subsidy_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Permission denied")

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        if subsidy["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Only pending subsidies can be approved"
            )

        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, {"status": "approved"}
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rejecting subsidy: {str(e)}"
        )


@subsidies_router.delete("{subsidy_id}")
def delete_subsidy(subsidy_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        role = user["role"]

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        if role == "admin":
            # Admin sets status to removed
            return DATABASES.update_document(
                DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, {"status": "removed"}
            )
        elif role == "provider" and subsidy["submitted_by"] == user["$id"]:
            # Provider sets status to withdrawn (only if in pending or approved state)
            if subsidy["status"] not in ["pending", "approved"]:
                raise HTTPException(
                    status_code=400,
                    detail="Only pending or approved subsidies can be withdrawn",
                )
            return DATABASES.update_document(
                DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, {"status": "withdrawn"}
            )
        else:
            raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error deleting subsidy: {str(e)}")


subsidy_requests_router = APIRouter(
    prefix="/subsidy_requests", tags=["Subsidy Requests"]
)

# Enums for status
REQUEST_STATUSES = ["requested", "accepted", "rejected", "withdrawn", "removed"]


# Pydantic models
class SubsidyRequestCreateModel(BaseModel):
    subsidy_id: str


def reject_pending_requests(subsidy_id: str):
    try:
        # Fetch all requests for the subsidy
        requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SUBSIDY_REQUESTS,
            queries=[Query.equal("subsidy_id", [subsidy_id])],
        )

        # Reject all pending requests
        for request in requests["documents"]:
            if request["status"] == "requested":
                DATABASES.update_document(
                    DATABASE_ID,
                    COLLECTION_SUBSIDY_REQUESTS,
                    request["$id"],
                    {"status": "rejected"},
                )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rejecting pending requests: {str(e)}"
        )


@subsidy_requests_router.get("")
def get_requests(
    email: Optional[str] = None,
    status: Optional[str] = "all",
    subsidy_id: Optional[str] = None,
):
    try:
        query_filters = []
        if status and status != "all":
            if status not in REQUEST_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {REQUEST_STATUSES}",
                )
            query_filters.append(Query.equal("status", [status]))
        if email:
            user = get_user_by_email_or_raise(email)
            role = user["role"]

            if role == "farmer":
                # Show requests sent by the farmer
                query_filters.append(Query.equal("farmer_id", [user["$id"]]))
            elif role == "provider":
                # Show requests on subsidies created by the provider
                provider_subsidies = DATABASES.list_documents(
                    DATABASE_ID,
                    COLLECTION_SUBSIDIES,
                    queries=[Query.equal("submitted_by", [user["$id"]])],
                )
                subsidy_ids = [
                    subsidy["$id"] for subsidy in provider_subsidies["documents"]
                ]
                if subsidy_ids:
                    query_filters.append(Query.contains("subsidy_id", subsidy_ids))

        if subsidy_id:
            query_filters.append(Query.equal("subsidy_id", [subsidy_id]))

        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, queries=query_filters
        )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error fetching subsidy requests: {str(e)}"
        )


@subsidy_requests_router.post("")
def create_request(data: SubsidyRequestCreateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "farmer":
            raise HTTPException(
                status_code=403, detail="Only farmers can create subsidy requests"
            )

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, data.subsidy_id, "Subsidy not found"
        )
        if subsidy["status"] != "approved":
            raise HTTPException(
                status_code=400, detail="Subsidy must be approved to create a request"
            )
        # only allow if farmer's address is in the subsidy locations
        farmer_location = user.get("address", "")
        if farmer_location not in subsidy["locations"]:
            raise HTTPException(
                status_code=400,
                detail="Farmer's address must be in the subsidy locations",
            )
        # Check if the farmer has already requested this subsidy
        # removed for testing
        existing_requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SUBSIDY_REQUESTS,
            queries=[
                Query.equal("farmer_id", [user["$id"]]),
                Query.equal("subsidy_id", [data.subsidy_id]),
            ],
        )
        if existing_requests["total"] > 0:
            raise HTTPException(
                status_code=400,
                detail="Farmer has already requested this subsidy",
            )
        request_data = {
            "subsidy_id": data.subsidy_id,
            "farmer_id": user["$id"],
            "status": "requested",
        }

        # Retry mechanism for ID uniqueness
        for attempt in range(3):
            try:
                return DATABASES.create_document(
                    DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, ID.unique(), request_data
                )
            except Exception as e:
                if "already exists" in str(e).lower() and attempt < 2:
                    continue  # Retry if ID conflict occurs
                raise HTTPException(
                    status_code=500, detail=f"Error creating subsidy request: {str(e)}"
                )
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error creating subsidy request: {str(e)}"
        )


@subsidy_requests_router.patch("{request_id}/accept")
def accept_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_SUBSIDY_REQUESTS, request_id, "Request not found"
        )

        if request["status"] != "requested":
            raise HTTPException(
                status_code=400,
                detail="Only requests with 'requested' status can be accepted",
            )

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, request["subsidy_id"], "Subsidy not found"
        )

        if user["role"] != "admin" and subsidy["submitted_by"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")

        # Update the request status
        DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request_id, {"status": "accepted"}
        )

        # Increment recipients_accepted in the subsidy
        if subsidy["status"] == "approved":
            updated_status = (
                "fulfilled"
                if subsidy["recipients_accepted"] + 1 == subsidy["max_recipients"]
                else "approved"
            )
            val = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_SUBSIDIES,
                subsidy["$id"],
                {
                    "recipients_accepted": subsidy["recipients_accepted"] + 1,
                    "status": updated_status,
                },
            )
            reject_pending_requests(subsidy["$id"])
            return val
        else:
            raise HTTPException(
                status_code=400,
                detail="Subsidy must be approved to accept a request",
            )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error accepting subsidy request: {str(e)}"
        )


@subsidy_requests_router.patch("{request_id}/reject")
def reject_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_SUBSIDY_REQUESTS, request_id, "Request not found"
        )

        if request["status"] != "requested":
            raise HTTPException(
                status_code=400,
                detail="Only requests with 'requested' status can be rejected",
            )

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, request["subsidy_id"], "Subsidy not found"
        )

        if user["role"] != "admin" and subsidy["submitted_by"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")

        # Update the request status
        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request_id, {"status": "rejected"}
        )

    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error rejecting subsidy request: {str(e)}"
        )


@subsidy_requests_router.delete("{request_id}")
def delete_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_SUBSIDY_REQUESTS, request_id, "Request not found"
        )
        if request["status"] not in ["requested", "accepted"]:
            raise HTTPException(
                status_code=400,
                detail="Only requests with 'requested' or 'accepted' status can be deleted",
            )

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, request["subsidy_id"], "Subsidy not found"
        )
        if request["farmer_id"] == user["$id"] and request["status"] == "requested":
            # Farmer withdraws the request
            return DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_SUBSIDY_REQUESTS,
                request_id,
                {"status": "withdrawn"},
            )
        elif user["role"] == "admin":
            # Admin removes the request
            updated = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_SUBSIDY_REQUESTS,
                request_id,
                {"status": "removed"},
            )

            # Update subsidy if the request was accepted
            if request["status"] == "accepted":
                if subsidy["status"] == "approved":
                    DATABASES.update_document(
                        DATABASE_ID,
                        COLLECTION_SUBSIDIES,
                        subsidy["$id"],
                        {"recipients_accepted": subsidy["recipients_accepted"] - 1},
                    )
                elif subsidy["status"] == "fulfilled":
                    DATABASES.update_document(
                        DATABASE_ID,
                        COLLECTION_SUBSIDIES,
                        subsidy["$id"],
                        {
                            "recipients_accepted": subsidy["recipients_accepted"] - 1,
                            "max_recipients": subsidy["max_recipients"] - 1,
                        },
                    )
            return updated
        else:
            raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error deleting subsidy request: {str(e)}"
        )


def test_subsidies():
    # Test case: Create a subsidy as admin
    subsidy_data = SubsidyCreateModel(
        type="crop",
        locations=["location1", "Farmer Address"],
        max_recipients=10,
        dynamic_fields='{"field1": "value1"}',
    )

    # Test case: Create a subsidy as provider
    print("\nTest: Create a subsidy as provider")
    try:
        subsidy_id = create_subsidy(subsidy_data, "provider@example.com")
        print(subsidy_id)
        subsidy_id = subsidy_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Create a subsidy as farmer
    print("\nTest: Create a subsidy as farmer")
    try:
        subsidy = create_subsidy(subsidy_data, "farmer@example.com")
        print(subsidy)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get subsidies as admin
    print("\nTest: Get subsidies as admin")
    try:
        subsidies = get_subsidies(email="admin@example.com")
        print(subsidies)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get subsidies as provider
    print("\nTest: Get subsidies as provider")
    try:
        subsidies = get_subsidies(email="provider@example.com")
        print(subsidies)
    except HTTPException as e:
        print(f"Error: {e.detail}")
        # Test case: Get subsidies as provider

    print("\nTest: Get subsidies as farmer")
    try:
        subsidies = get_subsidies(email="farmer@example.com")
        print(subsidies)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Approve subsidy
    print("\nTest: approve subsidy as admin")
    try:
        updated_subsidy = approve_subsidy(subsidy_id, "admin@example.com")
        print(updated_subsidy)
        subsidy_id_approved = updated_subsidy["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update subsidy as admin
    print("\nTest: Update subsidy as admin")
    update_data = SubsidyUpdateModel(
        max_recipients=15,
        location=["location3"],
        dynamic_fields='{"field2": "value2"}',
    )
    try:
        updated_subsidy = update_subsidy(
            subsidy_id_approved, update_data, "admin@example.com"
        )
        print(updated_subsidy)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject subsidy as admin
    print("\nTest: Reject subsidy as admin")
    try:
        subsidy_id = create_subsidy(subsidy_data, "provider@example.com")
        print(subsidy_id)
        subsidy_id = subsidy_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        rejected_subsidy = reject_subsidy(subsidy_id, "admin@example.com")
        print(rejected_subsidy)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete subsidy as admin
    print("\nTest: Delete subsidy as admin")
    try:
        deleted_subsidy = delete_subsidy(subsidy_id, "admin@example.com")
        print(deleted_subsidy)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete subsidy as provider
    print("\nTest: Delete subsidy as provider")
    try:
        subsidy_id = create_subsidy(subsidy_data, "provider@example.com")
        print(subsidy_id)
        subsidy_id = subsidy_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        deleted_subsidy = delete_subsidy(subsidy_id, "provider@example.com")
        print(deleted_subsidy)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Cleanup: Delete all subsidies
    print("\nCleanup: Delete all subsidies")
    try:
        subsidies = DATABASES.list_documents(DATABASE_ID, COLLECTION_SUBSIDIES)
        for subsidy in subsidies["documents"]:
            DATABASES.delete_document(DATABASE_ID, COLLECTION_SUBSIDIES, subsidy["$id"])
    except Exception as e:
        print(f"Error: {e}")


def test_requests():
    # Test case: Create a subsidy as provider for testing requests
    print("\nTest: Create a subsidy as provider for testing requests")
    subsidy_data = SubsidyCreateModel(
        type="crop",
        locations=["location1", "Farmer Address"],
        max_recipients=55,
        dynamic_fields='{"field1": "value1"}',
    )
    subsidy_id = None
    try:
        subsidy = create_subsidy(subsidy_data, "provider@example.com")
        print(subsidy)
        subsidy_id = subsidy["$id"]
        approve_subsidy(subsidy_id, "admin@example.com")
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Create a subsidy request as farmer
    print("\nTest: Create a subsidy request as farmer")
    request_id = None
    try:
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        request = create_request(request_data, "farmer@example.com")
        print(request)
        request_id = request["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

        # Test case: Create a subsidy request as farmer whose address is not in the subsidy locations
    print("\nTest: Create a subsidy request as farmer with location not in subsidy")
    try:
        # Create a new user with a different address
        farmerx = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmerx@example.com",
                "role": "farmer",
                "address": "Address",
                "name": "Farmer X",
            },
        )
        request = create_request(request_data, "farmerx@example.com")
        print(request)

    except HTTPException as e:
        print(f"Error: {e.detail}")
        # Cleanup: delete the created user
        DATABASES.delete_document(DATABASE_ID, COLLECTION_USERS, farmerx["$id"])

    # Test case: Create a subsidy request as non-farmer
    print("\nTest: Create a subsidy request as non-farmer")
    try:
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        print(create_request(request_data, "provider@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get requests as farmer
    print("\nTest: Get requests as farmer")
    try:
        requests = get_requests(email="farmer@example.com")
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get requests as provider
    print("\nTest: Get requests as provider")
    try:
        requests = get_requests(email="provider@example.com")
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get requests as admin
    print("\nTest: Get requests as admin")
    try:
        requests = get_requests(email="admin@example.com", status="requested")
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Get requests as farmer with subsidy_id
    print("\nTest: Get requests as farmer with subsidy_id")
    try:
        requests = get_requests(email="farmer@example.com", subsidy_id=subsidy_id)
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get requests as provider with subsidy_id
    print("\nTest: Get requests as provider with subsidy_id")
    try:
        requests = get_requests(email="provider@example.com", subsidy_id=subsidy_id)
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get requests as admin with subsidy_id
    print("\nTest: Get requests as admin with subsidy_id")
    try:
        requests = get_requests(
            email="admin@example.com", status="requested", subsidy_id=subsidy_id
        )
        print(requests)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Accept a request as admin
    print("\nTest: Accept a request as admin")
    try:
        accepted_request = accept_request(request_id, "admin@example.com")
        print(accepted_request)
        accepted_request_id = request_id
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Accept an already approved request as admin
    print("\nTest: Accept an already approved request as admin")
    try:
        accepted_request = accept_request(request_id, "admin@example.com")
        print(accepted_request)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Accept a request as the creator of the subsidy
    print("\nTest: Accept a request as the creator of the subsidy")
    try:
        # Create a new request for the subsidy
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        # create new user
        farmer = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer1@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 1",
            },
        )

        new_request = create_request(request_data, "farmer1@example.com")
        print(new_request)
        new_request_id = new_request["$id"]

        # Accept the request as the provider who created the subsidy
        accepted_request_by_creator = accept_request(
            new_request_id, "provider@example.com"
        )
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer["$id"]
        )  # Cleanup: delete the created user

        print(accepted_request_by_creator)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject a request as admin
    print("\nTest: Reject a request as admin")
    try:
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        farmer = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer2@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 2",
            },
        )
        new_request = create_request(request_data, "farmer2@example.com")
        print(new_request)
        new_request_id = new_request["$id"]
        rejected_request = reject_request(new_request_id, "admin@example.com")
        print(rejected_request)
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer["$id"]
        )  # Cleanup: delete the created user
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject a request as the creator of the subsidy
    print("\nTest: Reject a request as the creator of the subsidy")
    try:
        # Create a new request for the subsidy
        farmer = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer3@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 3",
            },
        )
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        new_request = create_request(request_data, "farmer3@example.com")
        print(new_request)
        new_request_id = new_request["$id"]

        # Reject the request as the provider who created the subsidy
        rejected_request_by_creator = reject_request(
            new_request_id, "provider@example.com"
        )
        print(rejected_request_by_creator)
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer["$id"]
        )  # Cleanup: delete the created user
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a request as farmer
    print("\nTest: Delete a request as farmer")
    try:
        farmer = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer4@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 3",
            },
        )
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        new_request = create_request(request_data, "farmer4@example.com")
        print(new_request)
        new_request_id = new_request["$id"]
        deleted_request = delete_request(new_request_id, "farmer4@example.com")
        print(deleted_request)
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer["$id"]
        )  # Cleanup: delete the created user
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete an accepted request as admin
    print("\nTest: Delete an accepted request as admin")
    try:
        deleted_request = delete_request(accepted_request_id, "admin@example.com")
        print(deleted_request)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Auto-reject pending requests when subsidy is fulfilled
    print("\nTest: Auto-reject pending requests when subsidy is fulfilled")
    try:
        # Create a new subsidy with max_recipients=1
        subsidy_data = SubsidyCreateModel(
            type="crop",
            locations=["location1", "Farmer Address"],
            max_recipients=1,
            dynamic_fields='{"field1": "value1"}',
        )
        subsidy = create_subsidy(subsidy_data, "provider@example.com")
        print(subsidy)
        subsidy_id = subsidy["$id"]
        approve_subsidy(subsidy_id, "admin@example.com")
        # Create two requests for the subsidy
        farmer1 = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer4@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 4",
            },
        )
        farmer2 = DATABASES.create_document(
            DATABASE_ID,
            COLLECTION_USERS,
            ID.unique(),
            {
                "email": "farmer5@example.com",
                "role": "farmer",
                "address": "Farmer Address",
                "name": "Farmer 5",
            },
        )
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        request_1 = create_request(request_data, "farmer4@example.com")
        print(request_1)
        request_2 = create_request(request_data, "farmer5@example.com")
        print(request_2)

        # Accept the first request
        accept_request(request_1["$id"], "admin@example.com")

        # Check the status of the second request
        rejected_request = DATABASES.get_document(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request_2["$id"]
        )
        print(
            f"Request 2 status after accepting Request 1: {rejected_request['status']}"
        )
        # Cleanup: delete the created users
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer1["$id"]
        )  # Cleanup: delete the created user
        DATABASES.delete_document(
            DATABASE_ID, COLLECTION_USERS, farmer2["$id"]
        )  # Cleanup: delete the created user
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Cleanup: Delete all subsidy requests and subsidies
    print("\nCleanup: Delete all subsidy requests and subsidies")
    try:
        requests = DATABASES.list_documents(DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS)
        for request in requests["documents"]:
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request["$id"]
            )
        subsidies = DATABASES.list_documents(DATABASE_ID, COLLECTION_SUBSIDIES)
        for subsidy in subsidies["documents"]:
            DATABASES.delete_document(DATABASE_ID, COLLECTION_SUBSIDIES, subsidy["$id"])
    except Exception as e:
        print(f"Error: {e}")
    # assuming only creator of subsidy and admin can acept, reject request
    # assuming delete request works only on requested and accepted status and accepted request can be deleted by admin only


def run_tests():
    test_subsidies()
    test_requests()
