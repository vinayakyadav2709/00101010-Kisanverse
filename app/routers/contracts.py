from fastapi import APIRouter, HTTPException
from core.config import (
    DATABASE_ID,
    DATABASES,
    COLLECTION_CONTRACTS,
    COLLECTION_CONTRACT_REQUESTS,
)
from core.dependencies import (
    get_user_by_email_or_raise,
    get_document_or_raise,
    get_state,
)
from pydantic import BaseModel, field_validator
from typing import List, Optional
from appwrite.query import Query
import json
from appwrite.id import ID

from datetime import datetime, timezone

contracts_router = APIRouter(prefix="/contracts", tags=["Contracts"])

# Enums for status
CONTRACT_STATUSES = ["listed", "accepted", "cancelled", "removed", "fulfilled"]


class ContractCreateModel(BaseModel):
    locations: Optional[List[str]] = ["all"]
    dynamic_fields: str
    crop_type: str
    quantity: int
    price_per_kg: float
    advance_payment: float
    delivery_date: str
    payment_terms: str

    @field_validator("dynamic_fields")
    @classmethod
    def validate_dynamic_fields(cls, value):
        try:
            json.loads(value)  # Ensure it's valid JSON
        except json.JSONDecodeError:
            raise ValueError("Invalid JSON format for dynamic_fields")
        return value

    @field_validator("delivery_date")
    @classmethod
    def validate_delivery_date(cls, value):
        try:
            delivery_date = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if delivery_date <= datetime.now():
                raise ValueError("Delivery date must be in the future")
        except ValueError:
            raise ValueError("Invalid date format. Use YYYY-MM-DD")
        return value


class ContractUpdateModel(BaseModel):
    dynamic_fields: Optional[str] = None
    crop_type: Optional[str] = None
    quantity: Optional[int] = None
    price_per_kg: Optional[float] = None
    advance_payment: Optional[float] = None
    delivery_date: Optional[str] = None
    payment_terms: Optional[str] = None

    @field_validator("dynamic_fields", mode="before")
    @classmethod
    def validate_dynamic_fields(cls, value):
        if value:
            try:
                json.loads(value)  # Ensure it's valid JSON
            except json.JSONDecodeError:
                raise ValueError("Invalid JSON format for dynamic_fields")
        return value

    @field_validator("delivery_date", mode="before")
    @classmethod
    def validate_delivery_date(cls, value):
        if value:
            try:
                delivery_date = datetime.fromisoformat(value.replace("Z", "+00:00"))
                if delivery_date <= datetime.now():
                    raise ValueError("Delivery date must be in the future")
            except ValueError:
                raise ValueError("Invalid date format. Use YYYY-MM-DD")
        return value


@contracts_router.get("")
def get_contracts(email: Optional[str] = None, status: Optional[str] = "all"):
    try:
        query_filters = []
        if status and status != "all":
            if status not in CONTRACT_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {CONTRACT_STATUSES} or 'all'",
                )
            query_filters.append(Query.equal("status", [status]))
        if email:
            user = get_user_by_email_or_raise(email)
            role = user["role"]

            if role == "buyer":
                # Show contracts created by the buyer
                query_filters.append(Query.equal("buyer_id", [user["$id"]]))
            elif role == "farmer":
                # Show contracts applicable to the farmer based on locations
                farmer_location = get_state(user["zipcode"])
                query_filters.append(
                    Query.or_queries(
                        [
                            Query.contains("locations", [farmer_location]),
                            Query.contains("locations", ["all"]),
                        ]
                    )
                )
        query_filters.append(Query.limit(10000))
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_CONTRACTS, queries=query_filters
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching contracts: {str(e)}"
        )


@contracts_router.post("")
def create_contract(data: ContractCreateModel, email: str):
    try:
        cleanup_expired_contracts()
        user = get_user_by_email_or_raise(email)
        if user["role"] != "buyer":
            raise HTTPException(
                status_code=403, detail="Only buyers can create contracts"
            )

        contract_data = data.model_dump()
        contract_data["buyer_id"] = user["$id"]
        contract_data["status"] = "listed"

        # Retry mechanism for ID uniqueness
        for attempt in range(3):
            try:
                return DATABASES.create_document(
                    DATABASE_ID, COLLECTION_CONTRACTS, ID.unique(), contract_data
                )
            except Exception as e:
                if "already exists" in str(e).lower() and attempt < 2:
                    continue  # Retry if ID conflict occurs
                raise HTTPException(
                    status_code=500, detail=f"Error creating contract: {str(e)}"
                )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error creating contract: {str(e)}"
        )


@contracts_router.patch("/{contract_id}")
def update_contract(contract_id: str, updates: ContractUpdateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "admin":
            raise HTTPException(status_code=403, detail="Permission denied")

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, contract_id, "Contract not found"
        )

        updated_data = updates.model_dump(exclude_unset=True)

        if contract["status"] not in ["listed", "accepted"]:
            raise HTTPException(
                status_code=400,
                detail="Status can only be updated if the contract is listed or accepted",
            )

        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_CONTRACTS, contract_id, updated_data
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error updating contract: {str(e)}"
        )


@contracts_router.delete("/{contract_id}")
def delete_contract(contract_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, contract_id, "Contract not found"
        )

        if user["role"] == "buyer" and contract["buyer_id"] == user["$id"]:
            # Buyer cancels the contract if it is listed
            if contract["status"] != "listed":
                raise HTTPException(
                    status_code=400,
                    detail="Only listed contracts can be cancelled by the buyer",
                )
            return DATABASES.update_document(
                DATABASE_ID, COLLECTION_CONTRACTS, contract_id, {"status": "cancelled"}
            )
        elif user["role"] == "admin":
            # Admin removes the contract if it is listed or accepted
            if contract["status"] not in ["listed", "accepted"]:
                raise HTTPException(
                    status_code=400,
                    detail="Only listed or accepted contracts can be removed by the admin",
                )
            return DATABASES.update_document(
                DATABASE_ID, COLLECTION_CONTRACTS, contract_id, {"status": "removed"}
            )
        else:
            raise HTTPException(status_code=403, detail="Permission denied")
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error deleting contract: {str(e)}"
        )


contract_requests_router = APIRouter(
    prefix="/contract_requests", tags=["Contract Requests"]
)

# Enums for status
REQUEST_STATUSES = [
    "pending",
    "accepted",
    "rejected",
    "withdrawn",
    "fulfilled",
    "removed",
]


# Pydantic models
class ContractRequestCreateModel(BaseModel):
    contract_id: str


def cleanup_expired_contracts():
    try:
        # Fetch all listed contracts
        listed_contracts = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_CONTRACTS,
            queries=[Query.equal("status", ["listed"]), Query.limit(10000)],
        )

        for contract in listed_contracts["documents"]:
            # Parse delivery_date using fromisoformat
            delivery_date = datetime.fromisoformat(
                contract["delivery_date"].replace("Z", "+00:00")
            )
            now = datetime.now(timezone.utc)  # Make datetime.now() offset-aware
            if delivery_date <= now:
                # Update contract status to removed
                DATABASES.update_document(
                    DATABASE_ID,
                    COLLECTION_CONTRACTS,
                    contract["$id"],
                    {"status": "removed"},
                )

                # Reject all pending requests for this contract
                reject_pending_requests(contract["$id"])

    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error during cleanup: {str(e)}")


def reject_pending_requests(contract_id: str):
    """
    Helper method to reject all pending requests for a given contract.
    """
    try:
        pending_requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_CONTRACT_REQUESTS,
            queries=[
                Query.equal("contract_id", [contract_id]),
                Query.equal("status", ["pending"]),
                Query.limit(10000),
            ],
        )
        for request in pending_requests["documents"]:
            DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_CONTRACT_REQUESTS,
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


@contract_requests_router.get("")
def get_requests(
    email: Optional[str] = None,
    status: Optional[str] = None,
    contract_id: Optional[str] = None,
):
    try:
        query_filters = []
        if status and status != "all":
            if status not in REQUEST_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid status. Must be one of {REQUEST_STATUSES} or 'all'",
                )
            query_filters.append(Query.equal("status", [status]))
        if email:
            user = get_user_by_email_or_raise(email)
            role = user["role"]

            if role == "farmer":
                # Show requests created by the farmer
                query_filters.append(Query.equal("farmer_id", [user["$id"]]))
            elif role == "buyer":
                # Show requests for contracts created by the buyer
                buyer_contracts = DATABASES.list_documents(
                    DATABASE_ID,
                    COLLECTION_CONTRACTS,
                    queries=[
                        Query.equal("buyer_id", [user["$id"]]),
                        Query.limit(10000),
                    ],
                )
                contract_ids = [
                    contract["$id"] for contract in buyer_contracts["documents"]
                ]
                query_filters.append(Query.equal("contract_id", contract_ids))

        if contract_id:
            query_filters.append(Query.equal("contract_id", [contract_id]))
        query_filters.append(Query.limit(10000))
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_CONTRACT_REQUESTS, queries=query_filters
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching contract requests: {str(e)}"
        )


@contract_requests_router.post("")
def create_request(data: ContractRequestCreateModel, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        if user["role"] != "farmer":
            raise HTTPException(
                status_code=403, detail="Only farmers can create contract requests"
            )

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, data.contract_id, "Contract not found"
        )
        if contract["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Contract must be listed to create a request"
            )

        # Ensure the delivery date is still valid
        delivery_date = datetime.fromisoformat(
            contract["delivery_date"].replace("Z", "+00:00")
        )
        now = datetime.now(timezone.utc)  # Make datetime.now() offset-aware
        if delivery_date <= now:
            raise HTTPException(
                status_code=400,
                detail="Cannot create a request for an expired contract",
            )
        # Ensure the farmer's location matches the contract's location
        farmer_location = get_state(user["zipcode"])
        if (
            farmer_location not in contract["locations"]
            and "all" not in contract["locations"]
        ):
            raise HTTPException(
                status_code=400,
                detail="Farmer's location does not match the contract's location",
            )

        # Ensure only one request per contract per farmer
        existing_requests = DATABASES.list_documents(
            DATABASE_ID,
            COLLECTION_CONTRACT_REQUESTS,
            queries=[
                Query.equal("contract_id", [data.contract_id]),
                Query.equal("farmer_id", [user["$id"]]),
            ],
        )
        if existing_requests["total"] > 0:
            raise HTTPException(
                status_code=400,
                detail="You have already created a request for this contract",
            )

        request_data = {
            "contract_id": data.contract_id,
            "farmer_id": user["$id"],
            "status": "pending",
        }

        # Retry mechanism for ID uniqueness
        for attempt in range(3):
            try:
                return DATABASES.create_document(
                    DATABASE_ID, COLLECTION_CONTRACT_REQUESTS, ID.unique(), request_data
                )
            except Exception as e:
                if "already exists" in str(e).lower() and attempt < 2:
                    continue  # Retry if ID conflict occurs
                raise HTTPException(
                    status_code=500, detail=f"Error creating contract request: {str(e)}"
                )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error creating contract request: {str(e)}"
        )


@contract_requests_router.patch("/{request_id}/accept")
def accept_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_CONTRACT_REQUESTS, request_id, "Request not found"
        )
        if request["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Only pending requests can be accepted"
            )

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, request["contract_id"], "Contract not found"
        )
        if user["role"] != "admin" and contract["buyer_id"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")
        if contract["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Contract must be listed to accept a request"
            )
        # Update the request status
        DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_CONTRACT_REQUESTS,
            request_id,
            {"status": "accepted"},
        )

        # Update the contract status
        d = DATABASES.update_document(
            DATABASE_ID, COLLECTION_CONTRACTS, contract["$id"], {"status": "accepted"}
        )

        # Reject all other pending requests for the contract
        reject_pending_requests(contract["$id"])
        return d

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error accepting contract request: {str(e)}"
        )


@contract_requests_router.patch("/{request_id}/reject")
def reject_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_CONTRACT_REQUESTS, request_id, "Request not found"
        )
        if request["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Only pending requests can be rejected"
            )

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, request["contract_id"], "Contract not found"
        )
        if user["role"] != "admin" and contract["buyer_id"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")

        # Update the request status
        return DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_CONTRACT_REQUESTS,
            request_id,
            {"status": "rejected"},
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error rejecting contract request: {str(e)}"
        )


@contract_requests_router.delete("/{request_id}")
def delete_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_CONTRACT_REQUESTS, request_id, "Request not found"
        )
        if request["status"] not in ["pending", "accepted"]:
            raise HTTPException(
                status_code=400,
                detail="Only requests with 'pending' or 'accepted' status can be deleted",
            )

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, request["contract_id"], "Contract not found"
        )

        if request["farmer_id"] == user["$id"]:
            # Farmer withdraws the request
            if request["status"] != "pending":
                raise HTTPException(
                    status_code=400,
                    detail="Only pending requests can be withdrawn by the farmer",
                )
            d = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_CONTRACT_REQUESTS,
                request_id,
                {"status": "withdrawn"},
            )
        elif user["role"] == "admin":
            # Admin removes the request
            d = DATABASES.update_document(
                DATABASE_ID,
                COLLECTION_CONTRACT_REQUESTS,
                request_id,
                {"status": "removed"},
            )
        else:
            raise HTTPException(status_code=403, detail="Permission denied")
        DATABASES.update_document(
            DATABASE_ID, COLLECTION_CONTRACTS, contract["$id"], {"status": "removed"}
        )
        return d
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error deleting contract request: {str(e)}"
        )


@contract_requests_router.patch("/{request_id}/fulfill")
def fulfill_request(request_id: str, email: str):
    try:
        user = get_user_by_email_or_raise(email)
        request = get_document_or_raise(
            COLLECTION_CONTRACT_REQUESTS, request_id, "Request not found"
        )
        if request["status"] != "accepted":
            raise HTTPException(
                status_code=400, detail="Only accepted requests can be fulfilled"
            )

        contract = get_document_or_raise(
            COLLECTION_CONTRACTS, request["contract_id"], "Contract not found"
        )
        if user["role"] != "admin" and contract["buyer_id"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")

        # Update the request status
        d = DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_CONTRACT_REQUESTS,
            request_id,
            {"status": "fulfilled"},
        )

        # Update the contract status
        DATABASES.update_document(
            DATABASE_ID, COLLECTION_CONTRACTS, contract["$id"], {"status": "fulfilled"}
        )

        return d
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fulfilling contract request: {str(e)}"
        )


def print_result(test_name, success, detail=None):
    if success:
        print(f"✔ {test_name}")
    else:
        print(f"✘ {test_name}")
        if detail:
            print(f"   Detail: {detail}")


def test_contracts():
    print("\nRunning Contract Tests...\n")

    # Test case: Create a contract as buyer
    test_name = "Create a contract as buyer"
    contract_data = ContractCreateModel(
        locations=["location1", "Farmer Address"],
        dynamic_fields='{"field1": "value1"}',
        crop_type="wheat",
        quantity=1000,
        price_per_kg=20.5,
        advance_payment=5000,
        delivery_date="2025-12-31",
        payment_terms="50% advance, 50% on delivery",
    )
    contract_id = None
    try:
        contract = create_contract(contract_data, "buyer@example.com")
        contract_id = contract["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Get contracts as buyer
    test_name = "Get contracts as buyer"
    try:
        contracts = get_contracts(email="buyer@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Update contract as admin
    test_name = "Update contract as admin"
    update_data = ContractUpdateModel(
        crop_type="rice",
        quantity=2000,
        price_per_kg=22.0,
        dynamic_fields='{"field2": "value2"}',
    )
    try:
        updated_contract = update_contract(
            contract_id, update_data, "admin@example.com"
        )
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Delete contract as buyer
    test_name = "Delete contract as buyer"
    try:
        deleted_contract = delete_contract(contract_id, "buyer@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)


def test_contract_requests():
    print("\nRunning Contract Request Tests...\n")

    # Test case: Create a contract as buyer
    test_name = "Create a contract as buyer"
    contract_data = ContractCreateModel(
        locations=["00000", "Farmer Address"],
        dynamic_fields='{"field1": "value1"}',
        crop_type="wheat",
        quantity=1000,
        price_per_kg=20.5,
        advance_payment=5000,
        delivery_date="2025-12-31",
        payment_terms="50% advance, 50% on delivery",
    )
    contract_id = None
    try:
        contract = create_contract(contract_data, "buyer@example.com")
        contract_id = contract["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Create a contract request as farmer
    test_name = "Create a contract request as farmer"
    request_id = None
    try:
        request_data = ContractRequestCreateModel(contract_id=contract_id)
        request = create_request(request_data, "farmer@example.com")
        request_id = request["$id"]
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Accept a contract request as admin
    test_name = "Accept a contract request as admin"
    try:
        accepted_request = accept_request(request_id, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Fulfill a contract request as admin
    test_name = "Fulfill a contract request as admin"
    try:
        fulfilled_request = fulfill_request(request_id, "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Reject a contract request as admin
    test_name = "Reject a contract request as admin"
    try:
        contract_data = ContractCreateModel(
            locations=["00000", "Farmer Address"],
            dynamic_fields='{"field1": "value1"}',
            crop_type="wheat",
            quantity=1000,
            price_per_kg=20.5,
            advance_payment=5000,
            delivery_date="2025-12-31",
            payment_terms="50% advance, 50% on delivery",
        )
        new_contract = create_contract(contract_data, "buyer@example.com")
        new_request_data = ContractRequestCreateModel(contract_id=new_contract["$id"])
        new_request = create_request(new_request_data, "farmer@example.com")
        rejected_request = reject_request(new_request["$id"], "admin@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)

    # Test case: Delete a contract request as farmer
    test_name = "Delete a contract request as farmer"
    try:
        contract_data = ContractCreateModel(
            locations=["00000", "Farmer Address"],
            dynamic_fields='{"field1": "value1"}',
            crop_type="wheat",
            quantity=1000,
            price_per_kg=20.5,
            advance_payment=5000,
            delivery_date="2025-12-31",
            payment_terms="50% advance, 50% on delivery",
        )
        new_contract = create_contract(contract_data, "buyer@example.com")
        new_request_data = ContractRequestCreateModel(contract_id=new_contract["$id"])
        new_request = create_request(new_request_data, "farmer@example.com")
        deleted_request = delete_request(new_request["$id"], "farmer@example.com")
        print_result(test_name, True)
    except HTTPException as e:
        print_result(test_name, False, e.detail)


def run_tests():
    test_contracts()
    test_contract_requests()


if __name__ == "__main__":
    run_tests()
