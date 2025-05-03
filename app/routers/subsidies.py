from fastapi import APIRouter, HTTPException

from core.config import (
    DATABASE_ID,
    DATABASES,
    COLLECTION_SUBSIDIES,
    COLLECTION_USERS,
    COLLECTION_SUBSIDY_REQUESTS,
)
from core.dependencies import (
    get_document_or_raise,
    get_user_by_email_or_raise,
    get_state,
)
from pydantic import BaseModel, field_validator
from typing import List, Optional, Union
from appwrite.query import Query
import json
from appwrite.id import ID

subsidies_router = APIRouter(prefix="/subsidies", tags=["Subsidies"])

# Enums for type and status
SUBSIDY_TYPES = ["cash", "asset", "training", "loan"]
SUBSIDY_STATUSES = ["listed", "removed", "fulfilled"]


def reject_pending_requests(subsidy_id: str):
    try:
        # Fetch all requests for the subsidy
        requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_SUBSIDY_REQUESTS,
            queries=[Query.equal("subsidy_id", [subsidy_id]), Query.limit(10000)],
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
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error rejecting pending requests: {str(e)}"
        )


# Pydantic models
class SubsidyCreateModel(BaseModel):
    type: str
    max_recipients: int
    dynamic_fields: str
    locations: List[str]
    provider: str
    program: str
    description: str
    eligibility: str
    benefits: str
    application_process: str

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

    @field_validator("locations")
    @classmethod
    def validate_locations(cls, value):
        if not value:
            raise ValueError("Locations cannot be empty")
        return value


class SubsidyUpdateModel(BaseModel):
    program: Optional[str] = None
    description: Optional[str] = None
    eligibility: Optional[str] = None
    type: Optional[str] = None
    benefits: Optional[str] = None
    application_process: Optional[str] = None
    dynamic_fields: Optional[str] = None
    max_recipients: Optional[int] = None
    provider: Optional[str] = None

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
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only admins can create subsidies.",
            )
        subsidy_data = data.model_dump()

        subsidy_data["status"] = "listed"  # Directly approve the subsidy

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
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error creating subsidy: {str(e)}")


@subsidies_router.get("")
def get_subsidies(
    email: str = None,
    type: Optional[str] = "all",
    status: Optional[str] = "all",
    provider: Optional[str] = None,
):
    try:
        query_filters = []

        # Apply type filter if not "all"
        if type != "all":
            if type not in SUBSIDY_TYPES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid type. Must be one of {SUBSIDY_TYPES} or 'all'.",
                )
            query_filters.append(Query.equal("type", [type]))

        # Apply status filter if not "all"
        if status != "all":
            if status not in SUBSIDY_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {SUBSIDY_STATUSES} or 'all'.",
                )
            query_filters.append(Query.equal("status", [status]))

        # Apply provider filter if specified
        if provider:
            query_filters.append(Query.equal("provider", [provider]))
        if email:
            user = get_user_by_email_or_raise(email)
            if user["role"] == "farmer":
                location = get_state(user["zipcode"])
                query_filters.append(
                    Query.or_queries(
                        [
                            Query.contains("locations", [location]),
                            Query.contains("locations", ["all"]),
                        ]
                    )
                )
        query_filters.append(Query.limit(10000))
        # Fetch subsidies with the applied filters
        subsidies = DATABASES.list_documents(
            DATABASE_ID, COLLECTION_SUBSIDIES, queries=query_filters
        )
        return subsidies
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching subsidies: {str(e)}"
        )


@subsidies_router.patch("/{subsidy_id}")
def update_subsidy(subsidy_id: str, updates: SubsidyUpdateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only admins can update subsidies.",
            )
        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        updated_data = updates.model_dump(exclude_unset=True)

        # Handle max_recipients updates
        if "max_recipients" in updated_data:
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

        val = DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, updated_data
        )
        if updated_data.get("status") == "fulfilled":
            # If the subsidy is fulfilled, reject all pending requests
            reject_pending_requests(subsidy["$id"])
        return val
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error updating subsidy: {str(e)}")


@subsidies_router.delete("/{subsidy_id}")
def delete_subsidy(subsidy_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only admins can delete subsidies.",
            )

        subsidy = get_document_or_raise(
            COLLECTION_SUBSIDIES, subsidy_id, "Subsidy not found"
        )

        val = DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDIES, subsidy_id, {"status": "removed"}
        )
        # Reject all pending requests for the removed subsidy
        reject_pending_requests(subsidy_id)
        return val
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting subsidy: {str(e)}")


subsidy_requests_router = APIRouter(
    prefix="/subsidy_requests", tags=["Subsidy Requests"]
)

# Enums for status
REQUEST_STATUSES = [
    "requested",
    "accepted",
    "rejected",
    "withdrawn",
    "removed",
    "fulfilled",
]


# Pydantic models
class SubsidyRequestCreateModel(BaseModel):
    subsidy_id: str


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
        if subsidy_id:
            query_filters.append(Query.equal("subsidy_id", [subsidy_id]))
        query_filters.append(Query.limit(10000))
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, queries=query_filters
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
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
        if subsidy["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Subsidy must be listed to create a request"
            )
        # only allow if farmer's zipcode is in the subsidy locations
        farmer_location = get_state(user["zipcode"])
        if (
            farmer_location not in subsidy["locations"]
            and "all" not in subsidy["locations"]
        ):
            raise HTTPException(
                status_code=400,
                detail="Farmer's zipcode must be in the subsidy locations",
            )
        # Check if the farmer has already requested this subsidy
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
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error creating subsidy request: {str(e)}"
        )


@subsidy_requests_router.patch("/{request_id}/accept")
def accept_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only admins can accept requests.",
            )

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

        # Update the request status
        DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request_id, {"status": "accepted"}
        )

        # Increment recipients_accepted in the subsidy
        if subsidy["status"] == "listed":
            updated_status = (
                "fulfilled"
                if subsidy["recipients_accepted"] + 1 == subsidy["max_recipients"]
                else "listed"
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
            if updated_status == "fulfilled":
                # If the subsidy is fulfilled, reject all pending requests
                reject_pending_requests(subsidy["$id"])
            return val
        else:
            raise HTTPException(
                status_code=400,
                detail="Subsidy must be listed to accept a request",
            )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error accepting subsidy request: {str(e)}"
        )


@subsidy_requests_router.patch("/{request_id}/reject")
def reject_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(
                status_code=403,
                detail="Permission denied. Only admins can reject requests.",
            )

        request = get_document_or_raise(
            COLLECTION_SUBSIDY_REQUESTS, request_id, "Request not found"
        )

        if request["status"] != "requested":
            raise HTTPException(
                status_code=400,
                detail="Only requests with 'requested' status can be rejected",
            )

        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_SUBSIDY_REQUESTS, request_id, {"status": "rejected"}
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error rejecting subsidy request: {str(e)}"
        )


@subsidy_requests_router.delete("/{request_id}")
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

        if user["role"] == "farmer" and request["farmer_id"] == user["$id"]:
            # Farmer withdraws the request
            val = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_SUBSIDY_REQUESTS,
                request_id,
                {"status": "withdrawn"},
            )
        elif user["role"] == "admin":
            # Admin removes the request
            val = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_SUBSIDY_REQUESTS,
                request_id,
                {"status": "removed"},
            )
        else:
            raise HTTPException(status_code=403, detail="Permission denied")
        return val
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error deleting subsidy request: {str(e)}"
        )


@subsidy_requests_router.patch("/{request_id}/fulfill")
def fulfill_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_SUBSIDY_REQUESTS, request_id, "Request not found"
        )

        # Check permissions: Only the farmer who created the request or an admin can fulfill it
        if user["role"] != "admin" and request["farmer_id"] != user["$id"]:
            raise HTTPException(
                status_code=403,
                detail="Only the farmer who created the request or an admin can fulfill requests",
            )

        # Check if the request is in the correct status
        if request["status"] != "accepted":
            raise HTTPException(
                status_code=400, detail="Request not available for fulfillment"
            )

        # Update request status to fulfilled
        return DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_SUBSIDY_REQUESTS,
            request_id,
            {"status": "fulfilled"},
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fulfilling subsidy request: {str(e)}"
        )


def print_result(test_name, success, detail=None):
    if success:
        print(f"✔ {test_name}")
    else:
        print(f"✘ {test_name}")
        if detail:
            print(f"   Detail: {detail}")


def test_subsidies():
    print("\nRunning Subsidy Tests...\n")

    # Test case: Create a subsidy as admin
    test_name = "Create a subsidy as admin"
    subsidy_data = SubsidyCreateModel(
        program="Crop Insurance",
        description="Insurance for crop damage",
        eligibility="Farmers with valid land records",
        type="cash",
        benefits="5000",
        application_process="Apply online",
        locations=["00000", "12345"],
        dynamic_fields='{"field1": "value1"}',
        max_recipients=10,
        provider="Organization A",
    )
    subsidy_id = None
    try:
        subsidy = create_subsidy(subsidy_data, "admin@example.com")
        subsidy_id = subsidy["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Get subsidies with no filters
    test_name = "Get subsidies with no filters"
    try:
        subsidies = get_subsidies()
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Update subsidy as admin
    test_name = "Update subsidy as admin"
    update_data = SubsidyUpdateModel(
        max_recipients=15,
        locations=["location3"],
        dynamic_fields='{"field2": "value2"}',
    )
    try:
        updated_subsidy = update_subsidy(subsidy_id, update_data, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Delete subsidy as admin
    test_name = "Delete subsidy as admin"
    try:
        deleted_subsidy = delete_subsidy(subsidy_id, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)


def test_requests():
    print("\nRunning Subsidy Request Tests...\n")

    # Test case: Create a subsidy as admin for testing requests
    test_name = "Create a subsidy as admin for testing requests"
    subsidy_data = SubsidyCreateModel(
        program="Crop Insurance",
        description="Insurance for crop damage",
        eligibility="Farmers with valid land records",
        type="cash",
        benefits="5000",
        application_process="Apply online",
        locations=["00000", "Farmer Address"],
        dynamic_fields='{"field1": "value1"}',
        max_recipients=5,
        provider="Organization A",
    )
    subsidy_id = None
    try:
        subsidy = create_subsidy(subsidy_data, "admin@example.com")
        subsidy_id = subsidy["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Create a subsidy request as farmer
    test_name = "Create a subsidy request as farmer"
    request_id = None
    try:
        request_data = SubsidyRequestCreateModel(subsidy_id=subsidy_id)
        request = create_request(request_data, "farmer@example.com")
        request_id = request["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Accept a request as admin
    test_name = "Accept a request as admin"
    try:
        accept_request(request_id, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Fulfill a request as farmer
    test_name = "Fulfill a request as farmer"
    try:
        fulfilled_request = fulfill_request(request_id, "farmer@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Cleanup: Delete all subsidy requests and subsidies
    test_name = "Cleanup: Delete all subsidy requests and subsidies"
    try:
        delete_subsidy(subsidy_id, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)


def run_tests():
    test_subsidies()
    test_requests()
