from fastapi import APIRouter, HTTPException
from appwrite.id import ID
from core.config import (
    DATABASE_ID,
    DATABASES,
    COLLECTION_CROP_LISTINGS,
    COLLECTION_BIDS,
)
from core.dependencies import (
    get_user_by_email,
    get_user_by_email_or_raise,
    get_document_or_raise,
)
from typing import List, Optional
from pydantic import BaseModel
from appwrite.query import Query
from datetime import datetime, timezone

listings_router = APIRouter(prefix="/listing", tags=["Listings"])

# only lisiting implements admin thing. rest dont


def reject_pending_bids(listing_id: str):
    try:
        # Fetch all bids for the given listing_id with 'pending' status
        query = [
            Query.equal("listing_id", [listing_id]),
            Query.equal("status", ["pending"]),
            Query.limit(10000),
        ]
        pending_bids = DATABASES.list_documents(
            DATABASE_ID, COLLECTION_BIDS, queries=query
        )["documents"]

        if not pending_bids:
            return {"message": "No pending bids to reject."}

        # Update all pending bids to rejected
        updated_bids = []
        for bid in pending_bids:
            bid_id = bid["$id"]
            updated_bids.append(
                DATABASES.update_document(
                    DATABASE_ID, COLLECTION_BIDS, bid_id, {"status": "rejected"}
                )
            )

        # Wait for all updates to complete (if needed, but for simplicity assuming synchronous)
        for update in updated_bids:
            update

        return {"message": f"{len(pending_bids)} pending bids have been rejected."}

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error rejecting pending bids: {str(e)}"
        )


def reject_bids_below_available_quantity(listing_id: str, available_quantity: float):
    try:
        # Fetch all bids for the given listing_id with 'pending' status
        query = [
            Query.equal("listing_id", [listing_id]),
            Query.equal("status", ["pending"]),
            Query.limit(10000),
        ]
        pending_bids = DATABASES.list_documents(
            DATABASE_ID, COLLECTION_BIDS, queries=query
        )["documents"]

        if not pending_bids:
            return {"message": "No pending bids to reject."}

        # Update all pending bids to rejected
        updated_bids = []
        for bid in pending_bids:
            if bid["quantity"] > available_quantity:
                bid_id = bid["$id"]
                updated_bids.append(
                    DATABASES.update_document(
                        DATABASE_ID, COLLECTION_BIDS, bid_id, {"status": "rejected"}
                    )
                )

        # Wait for all updates to complete (if needed, but for simplicity assuming synchronous)
        for update in updated_bids:
            update

        return {
            "message": f"{len(updated_bids)} bids have been rejected due to insufficient quantity."
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error rejecting bids: {str(e)}")


# Pydantic models for request bodies
class CropListingCreateModel(BaseModel):
    crop_type: str
    price_per_kg: float
    total_quantity: float


class CropListingUpdateModel(BaseModel):
    crop_type: Optional[str] = None
    price_per_kg: Optional[float] = None
    total_quantity: Optional[float] = None


class BidCreateModel(BaseModel):
    quantity: Optional[float] = None
    price_per_kg: Optional[float] = None
    listing_id: str


class BidUpdateModel(BaseModel):
    quantity: Optional[float]
    price_per_kg: Optional[float]
    listing_id: str


# Listing routes
class CropListingFilterModel(BaseModel):
    email: Optional[str] = None  # email of the farmer (for admin filtering)
    type: Optional[str] = "all"  # 'all', 'listed', 'removed', 'cancelled', 'fulfilled'
    requester_email: str  # email of the requester (admin or farmer)


LISTING_STATUSES = ["listed", "removed", "cancelled", "fulfilled"]


@listings_router.get("")
def get_crop_listings(email: Optional[str] = None, type: str = "all"):
    try:
        query_filters = []

        if email:
            farmer = get_user_by_email_or_raise(email)
            if farmer["role"] != "farmer":
                raise HTTPException(
                    status_code=403, detail="Only farmers can have listings"
                )
            query_filters.append(Query.equal("farmer_id", [farmer["$id"]]))

        if type != "all":
            if type not in LISTING_STATUSES:
                raise HTTPException(status_code=400, detail="Invalid listing status")
            query_filters.append(Query.equal("status", [type]))
        query_filters.append(Query.limit(10000))
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_CROP_LISTINGS, queries=query_filters
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error fetching crop listings: {str(e)}"
        )


@listings_router.get("/{listing_id}")
def get_crop_listing(listing_id: str):
    try:
        return get_document_or_raise(
            COLLECTION_CROP_LISTINGS, listing_id, "Crop listing not found"
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error retrieving crop listing: {str(e)}"
        )


@listings_router.post("")
def create_crop_listing(data: CropListingCreateModel, email: str):
    try:
        user = get_user_by_email(email)
        role = user["role"]
        if role != "farmer":
            raise HTTPException(
                status_code=403, detail="Only farmers can create listings"
            )

        crop_listing_data = data.model_dump()
        if (
            crop_listing_data["total_quantity"] <= 0
            or crop_listing_data["price_per_kg"] <= 0
        ):
            raise HTTPException(
                status_code=400,
                detail="Invalid data: total_quantity and price_per_kg must be positive",
            )
        crop_listing_data["farmer_id"] = user["$id"]
        crop_listing_data["available_quantity"] = crop_listing_data["total_quantity"]
        now_utc = datetime.now(timezone.utc).isoformat()
        crop_listing_data["added_at"] = now_utc
        crop_listing_data["updated_at"] = now_utc
        crop_listing_data["total_quantity"] = round(
            crop_listing_data["total_quantity"], 2
        )
        crop_listing_data["price_per_kg"] = round(crop_listing_data["price_per_kg"], 2)
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                unique_id = ID.unique()
                return DATABASES.create_document(
                    DATABASE_ID, COLLECTION_CROP_LISTINGS, unique_id, crop_listing_data
                )
            except Exception as e:
                if "already exists" in str(e) and attempt < max_attempts - 1:
                    continue  # Retry with a new unique ID
                raise HTTPException(
                    status_code=500,
                    detail=f"Error creating crop listing: {str(e)}",
                )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error creating crop listing: {str(e)}"
        )


@listings_router.patch("/{listing_id}")
def update_crop_listing(listing_id: str, updates: CropListingUpdateModel, email: str):
    try:
        user = get_user_by_email(email)
        role = user["role"]

        listing = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, listing_id, "Listing not found"
        )
        is_owner = listing["farmer_id"] == user["$id"]
        is_closed = listing["available_quantity"] == 0
        if listing["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Cannot update a non-listed crop listing"
            )
        if is_owner and updates.get("crop_type") != listing["crop_type"]:
            raise HTTPException(
                status_code=400,
                detail="Only admin can change crop type. Please contact admin.",
            )
        if role != "admin" and not is_owner:
            raise HTTPException(status_code=403, detail="Permission denied")

        # Prepare updates, excluding None or empty string
        updated_data = {}
        for key, value in updates.model_dump(exclude_unset=True).items():
            if value is not None and (
                not isinstance(value, str) or value.strip() != ""
            ):
                updated_data[key] = value

        # Validate status update

        # Validate total_quantity update
        if "total_quantity" in updated_data:
            if is_closed:
                raise HTTPException(
                    status_code=403,
                    detail="Cannot update total_quantity for a closed listing",
                )

            current_total = listing["total_quantity"]
            current_available = listing["available_quantity"]
            new_total = updated_data["total_quantity"]
            quantity_diff = new_total - current_total

            new_available = current_available + quantity_diff
            if new_available < 0:
                raise HTTPException(
                    status_code=400,
                    detail=f"Update invalid. Available quantity would go below zero by {abs(new_available)}.",
                )
            updated_data["available_quantity"] = new_available

        # Auto-close logic
        if (
            "available_quantity" in updated_data
            and updated_data["available_quantity"] == 0
        ):
            updated_data["status"] = "fulfilled"

        # Final timestamp
        updated_data["updated_at"] = datetime.now(timezone.utc).isoformat()
        updated_data["total_quantity"] = round(
            updated_data.get("total_quantity", listing["total_quantity"]), 2
        )
        updated_data["price_per_kg"] = round(
            updated_data.get("price_per_kg", listing["price_per_kg"]), 2
        )
        output = DATABASES.update_document(
            DATABASE_ID, COLLECTION_CROP_LISTINGS, listing_id, updated_data
        )
        if updated_data.get("status") == "fulfilled":
            reject_pending_bids(listing_id)
        else:
            reject_bids_below_available_quantity(
                listing_id,
                updated_data.get(
                    "available_quantity" or listing["available_quantity"],
                    listing["available_quantity"],
                ),
            )
        return output
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error updating crop listing: {str(e)}"
        )


@listings_router.delete("/{listing_id}")
def cancel_crop_listing(listing_id: str, email: str):
    try:
        user = get_user_by_email(email)
        role = user["role"]

        doc = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, listing_id, "Listing not found"
        )
        if role != "admin" and doc["farmer_id"] != user["$id"]:
            raise HTTPException(status_code=403, detail="Permission denied")
        status = "removed" if role == "admin" else "cancelled"
        if doc["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Cannot cancel a non-listed crop listing"
            )
        updated_listing = DATABASES.update_document(
            DATABASE_ID, COLLECTION_CROP_LISTINGS, listing_id, {"status": status}
        )

        reject_pending_bids(listing_id)
        return updated_listing
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(
            status_code=500, detail=f"Error canceling crop listing: {str(e)}"
        )


# Bid routes
bids_router = APIRouter(prefix="/bids", tags=["Listings"])


@bids_router.post("")
def place_bid(data: BidCreateModel, email: str):
    try:
        user = get_user_by_email(email)
        bid_data = data.model_dump()
        listing = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, bid_data["listing_id"], "Listing not found"
        )
        if user["role"] != "buyer":
            raise HTTPException(status_code=403, detail="Only buyers can place bids")
        if not bid_data["quantity"]:
            # set from listing
            bid_data["quantity"] = listing["available_quantity"]
        if not bid_data["price_per_kg"]:
            # set from listing
            bid_data["price_per_kg"] = listing["price_per_kg"]
        if bid_data["quantity"] <= 0 or bid_data["price_per_kg"] <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid data: quantity and price_per_kg must be positive",
            )
        # Check if the listing exists and is active
        if not listing or listing["status"] != "listed":
            raise HTTPException(status_code=404, detail="Listing not found or inactive")
        if listing["available_quantity"] < bid_data["quantity"]:
            raise HTTPException(
                status_code=400,
                detail="Bid quantity exceeds available quantity in the listing",
            )
        bid_data["buyer_id"] = user["$id"]
        bid_data["timestamp"] = datetime.now(timezone.utc).isoformat()
        data.quantity = round(data.quantity, 2)
        data.price_per_kg = round(data.price_per_kg, 2)
        return DATABASES.create_document(
            DATABASE_ID, COLLECTION_BIDS, ID.unique(), bid_data
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error placing bid: {str(e)}")


BID_STATUSES = [
    "pending",
    "accepted",
    "rejected",
    "withdrawn",
    "fulfilled",
    "removed",
]


@bids_router.get("")
def get_bids(
    email: Optional[str] = None,
    type: Optional[str] = "all",
    listing_id: Optional[str] = None,
):
    try:
        query_filters = []

        # Filter by type
        if type and type != "all":
            if type not in BID_STATUSES:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid bid status. Must be one of 'pending', 'accepted', 'rejected', 'withdrawn', 'fulfilled', 'removed', or 'all'.",
                )
            query_filters.append(Query.equal("status", [type]))

        # Filter by email
        if email:
            user = get_user_by_email_or_raise(email)
            role = user["role"]
            if role == "buyer":
                # Return all bids placed by the buyer
                query_filters.append(Query.equal("buyer_id", [user["$id"]]))
            elif role == "farmer":
                # Return all bids on listings created by the farmer
                farmer_listings = DATABASES.list_documents(
                    DATABASE_ID,
                    COLLECTION_CROP_LISTINGS,
                    queries=[
                        Query.equal("farmer_id", [user["$id"]]),
                        Query.limit(10000),
                    ],
                )
                listing_ids = [
                    listing["$id"] for listing in farmer_listings["documents"]
                ]
                if listing_ids:
                    query_filters.append(Query.equal("listing_id", listing_ids))

        # Filter by listing_id
        if listing_id:
            query_filters.append(Query.equal("listing_id", [listing_id]))
        query_filters.append(Query.limit(10000))
        # Fetch and return bids
        return DATABASES.list_documents(
            DATABASE_ID, COLLECTION_BIDS, queries=query_filters
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching bids: {str(e)}")


@bids_router.get("/{bid_id}")
def get_specific_bid(bid_id: str):
    try:
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        return bid
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fetching bid: {str(e)}")


@bids_router.patch("/{bid_id}/accept")
def accept_bid(bid_id: str, email: str):
    try:
        user = get_user_by_email(email)
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        listing = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, bid["listing_id"], "Listing not found"
        )

        if listing["farmer_id"] != user["$id"]:
            raise HTTPException(
                status_code=403, detail="Only the farmer who created can accept bids"
            )
        if bid["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Bid not available for acceptance"
            )

        if listing["available_quantity"] < bid["quantity"]:
            raise HTTPException(
                status_code=400, detail="Not enough available quantity in the listing"
            )

        # Update bid status to accepted
        bid_updated = DATABASES.update_document(
            DATABASE_ID, COLLECTION_BIDS, bid_id, {"status": "accepted"}
        )

        # Update the listing's available quantity
        new_available_quantity = listing["available_quantity"] - bid["quantity"]
        status = "fulfilled" if new_available_quantity == 0 else listing["status"]
        DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_CROP_LISTINGS,
            listing["$id"],
            {
                "available_quantity": new_available_quantity,
                "status": status,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        if status == "fulfilled":
            reject_pending_bids(listing["$id"])
        else:
            reject_bids_below_available_quantity(listing["$id"], new_available_quantity)
        return bid_updated
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error accepting bid: {str(e)}")


@bids_router.patch("/{bid_id}/reject")
def reject_bid(bid_id: str, email: str):
    try:
        user = get_user_by_email(email)
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        listing = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, bid["listing_id"], "Listing not found"
        )
        if not listing:
            raise HTTPException(status_code=404, detail="Listing not found")
        if listing["farmer_id"] != user["$id"]:
            raise HTTPException(
                status_code=403, detail="Only the farmer who created can reject bids"
            )
        if bid["status"] != "pending":
            raise HTTPException(
                status_code=400, detail="Bid not available for rejection"
            )

        # Update bid status to rejected
        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_BIDS, bid_id, {"status": "rejected"}
        )

    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error rejecting bid: {str(e)}")


@bids_router.patch("/{bid_id}")
def update_bid(bid_id: str, updates: BidUpdateModel, email: str):
    try:
        user = get_user_by_email(email)
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        listing = get_document_or_raise(
            COLLECTION_CROP_LISTINGS, bid["listing_id"], "Listing not found"
        )

        if bid["buyer_id"] != user["$id"] and user["role"] != "admin":
            raise HTTPException(
                status_code=403, detail="Only the buyer can update their bid"
            )

        if bid["status"] not in ["pending", "accepted"]:
            raise HTTPException(
                status_code=400, detail="Bid can only be updated if pending or accepted"
            )
        if bid["status"] == "accepted" and user["role"] != "admin":
            raise HTTPException(
                status_code=400, detail="Accepted bid can only be updated by admin"
            )
        updated_data = updates.model_dump(exclude_unset=True)
        if listing["status"] != "listed":
            raise HTTPException(
                status_code=400, detail="Cannot update a non-listed crop listing"
            )
        # If the bid quantity or price is updated, validate
        if "quantity" in updated_data and updated_data["quantity"] <= 0:
            raise HTTPException(
                status_code=400, detail="Quantity must be greater than zero"
            )

        if "price_per_kg" in updated_data and updated_data["price_per_kg"] <= 0:
            raise HTTPException(
                status_code=400, detail="Price must be greater than zero"
            )

        # Calculate the new available quantity in the listing
        new_available_quantity = (
            listing["available_quantity"]
            - bid["quantity"]
            + updated_data.get("quantity", bid["quantity"])
        )

        if new_available_quantity < 0:
            raise HTTPException(
                status_code=400,
                detail="Bid quantity exceeds available quantity in the listing",
            )

        updated_data["price_per_kg"] = round(
            updated_data.get("price_per_kg", bid["price_per_kg"]), 2
        )
        updated_data["quantity"] = round(
            updated_data.get("quantity", bid["quantity"]), 2
        )

        # Update bid
        updated_bid = DATABASES.update_document(
            DATABASE_ID, COLLECTION_BIDS, bid_id, updated_data
        )
        # Update the listing's available quantity
        DATABASES.update_document(
            DATABASE_ID,
            COLLECTION_CROP_LISTINGS,
            listing["$id"],
            {
                "available_quantity": new_available_quantity,
                "status": "fulfilled"
                if new_available_quantity == 0
                else listing["status"],
            },
        )
        # Check if the updated quantity fulfills the listing
        if new_available_quantity == 0:
            reject_pending_bids(listing["$id"])
        else:
            reject_bids_below_available_quantity(
                listing["$id"],
                new_available_quantity,
            )
        return updated_bid
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error updating bid: {str(e)}")


@bids_router.delete("/{bid_id}")
def delete_bid(bid_id: str, email: str):
    try:
        user = get_user_by_email(email)
        role = user["role"]
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        # Only the buyer or admin can delete the bid
        if role != "admin" and bid["buyer_id"] != user["$id"]:
            raise HTTPException(
                status_code=403,
                detail="Permission denied: Only the buyer or admin can delete the bid",
            )
        if bid["status"] not in ["pending", "accepted"]:
            raise HTTPException(
                status_code=400, detail="Bid can only be deleted if pending or accepted"
            )
        if bid["status"] == "accepted" and role != "admin":
            # If the bid is accepted, only admin can delete it
            raise HTTPException(
                status_code=400,
                detail="Cannot delete an accepted bid. Please contact the admin.",
            )

        new_status = "removed" if role == "admin" else "withdrawn"

        updated_data = {
            "status": new_status,
        }

        bid_updated = DATABASES.update_document(
            DATABASE_ID, COLLECTION_BIDS, bid_id, updated_data
        )
        if bid["status"] == "accepted" and new_status == "removed":
            # If the bid is accepted and removed, update the listing's available quantity or total quantity
            # if listing not listed, then reduce the total quantity else update available
            listing = get_document_or_raise(
                COLLECTION_CROP_LISTINGS, bid["listing_id"], "Listing not found"
            )
            if listing["status"] == "listed":
                new_available_quantity = listing["available_quantity"] + bid["quantity"]
                DATABASES.update_document(
                    DATABASE_ID,
                    COLLECTION_CROP_LISTINGS,
                    listing["$id"],
                    {
                        "available_quantity": new_available_quantity,
                        "status": "listed",
                    },
                )
            else:
                new_total_quantity = listing["total_quantity"] - bid["quantity"]
                DATABASES.update_document(
                    DATABASE_ID,
                    COLLECTION_CROP_LISTINGS,
                    listing["$id"],
                    {
                        "total_quantity": new_total_quantity,
                    },
                )
        return bid_updated
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error deleting bid: {str(e)}")


# bid fulfilled. set by admin or creator of the bid i.e buyer can
@bids_router.patch("/{bid_id}/fulfill")
def fulfill_bid(bid_id: str, email: str):
    try:
        user = get_user_by_email(email)
        bid = get_document_or_raise(COLLECTION_BIDS, bid_id, "Bid not found")
        if user["role"] != "admin" and bid["buyer_id"] != user["$id"]:
            raise HTTPException(
                status_code=403,
                detail="Only the buyer who created or admin can fulfill bids",
            )
        if bid["status"] != "accepted":
            raise HTTPException(
                status_code=400, detail="Bid not available for fulfillment"
            )

        # Update bid status to fulfilled
        return DATABASES.update_document(
            DATABASE_ID, COLLECTION_BIDS, bid_id, {"status": "fulfilled"}
        )
    except Exception as e:
        if isinstance(e, HTTPException):
            # If it's already an HTTPException, return it as is
            raise e
        raise HTTPException(status_code=500, detail=f"Error fulfilling bid: {str(e)}")


def test_listings():
    # Test case: Create a valid crop listing
    crop_listing_id = None
    print("Test: Create a valid crop listing")
    listing_data = CropListingCreateModel(
        crop_type="Wheat",
        price_per_kg=20.0,
        total_quantity=100.0,
    )
    try:
        crop_listing_id = create_crop_listing(listing_data, "farmer@example.com")
        print(crop_listing_id)
        crop_listing_id = crop_listing_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Create a crop listing with invalid data
    print("\nTest: Create a crop listing with invalid data")
    listing_data.total_quantity = -10.0
    try:
        print(create_crop_listing(listing_data, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get crop listings without email
    print("\nTest: Get crop listings without email")
    try:
        # Get all crop listings (no filters)
        print(get_crop_listings())

        # Get crop listings filtered by type (listed)
        print(get_crop_listings(type="listed"))

        # Get crop listings filtered by type (removed)
        print(get_crop_listings(type="removed"))

        # Get crop listings filtered by type (fulfilled)
        print(get_crop_listings(type="fulfilled"))

        # Get crop listings filtered by type (cancelled)
        print(get_crop_listings(type="cancelled"))

        # Get crop listings with invalid type
        print(get_crop_listings(type="invalid_type"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get crop listings with email
    print("\nTest: Get crop listings with email")
    try:
        # Get crop listings filtered by farmer email
        print(get_crop_listings(email="farmer@example.com"))

        # Get crop listings filtered by farmer email and type (listed)
        print(get_crop_listings(email="farmer@example.com", type="listed"))

        # Get crop listings filtered by farmer email and type (removed)
        print(get_crop_listings(email="farmer@example.com", type="removed"))

        # Get crop listings filtered by farmer email and type (fulfilled)
        print(get_crop_listings(email="farmer@example.com", type="fulfilled"))

        # Get crop listings filtered by farmer email and type (cancelled)
        print(get_crop_listings(email="farmer@example.com", type="cancelled"))

        # Get crop listings filtered by farmer email with invalid type
        print(get_crop_listings(email="farmer@example.com", type="invalid_type"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get crop listings for non farmer
    print("\nTest: Get crop listings for non farmer")
    try:
        # Get crop listings filtered by farmer email
        print(get_crop_listings(email="admin@example.com"))

    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get a crop listing
    print("\nTest: Get a crop listing")
    try:
        print(get_crop_listing(crop_listing_id))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get a non existing crop listing
    print("\nTest: Get a non existing crop listing")
    try:
        print(get_crop_listing("non_existing_listing_id"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a crop listing as the owner
    print("\nTest: Update a crop listing as the owner")
    update_data = CropListingUpdateModel(
        price_per_kg=25.0,
        total_quantity=120.0,
    )
    try:
        print(update_crop_listing(crop_listing_id, update_data, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a crop listing as admin
    print("\nTest: Update a crop listing as admin")
    try:
        print(update_crop_listing(crop_listing_id, update_data, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a crop listing as buyer
    print("\nTest: Update a crop listing as buyer")
    try:
        print(update_crop_listing(crop_listing_id, update_data, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a non existent crop listing
    print("\nTest: Update a non existent crop listing")
    try:
        print(
            update_crop_listing(
                "non_existant_listing", update_data, "admin@example.com"
            )
        )
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a crop listing with invalid data
    print("\nTest: Update a crop listing with invalid data")
    update_data = CropListingUpdateModel(
        price_per_kg=0,
        total_quantity=0,
    )
    try:
        print(update_crop_listing(crop_listing_id, update_data, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Cancel a crop listing as buyer
    print("\nTest: Cancel a crop listing as buyer")
    try:
        print(cancel_crop_listing(crop_listing_id, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Cancel a crop listing as the owner
    print("\nTest: Cancel a crop listing as the owner")
    try:
        print(cancel_crop_listing(crop_listing_id, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Cancel a crop listing as admin
    print("\nTest: Cancel a crop listing as admin")
    try:
        listing_data = CropListingCreateModel(
            crop_type="Wheat",
            price_per_kg=20.0,
            total_quantity=100.0,
        )
        crop_listing_id = create_crop_listing(listing_data, "farmer@example.com")
        print(crop_listing_id)
        crop_listing_id = crop_listing_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(cancel_crop_listing(crop_listing_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Cancel a non existent crop listing
    print("\nTest: Cancel a non existent crop listing")
    try:
        print(cancel_crop_listing("non_existant_listing", "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
        print("\nDelete all bids and listings")
    try:
        # delete all listings
        listings = DATABASES.list_documents(DATABASE_ID, COLLECTION_CROP_LISTINGS)
        for listing in listings["documents"]:
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_CROP_LISTINGS, listing["$id"]
            )
    except Exception as e:
        print(f"Error: {e.detail}")


def test_bids():
    # create a crop listing
    crop_listing_id = None
    bid_id = None
    print("Create a crop listing for testing bids")
    try:
        listing_data = CropListingCreateModel(
            crop_type="Wheat",
            price_per_kg=20.0,
            total_quantity=1000.0,
        )
        crop_listing_id = create_crop_listing(listing_data, "farmer@example.com")
        print("Crop listing created")
        crop_listing_id = crop_listing_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Place a valid bid
    print("Test: Place a valid bid")
    bid_data = BidCreateModel(
        quantity=10.0,
        listing_id=crop_listing_id,
        price_per_kg=20.0,
    )
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Place a bid with invalid quantity
    print("\nTest: Place a bid with invalid quantity")
    bid_data.quantity = -5.0
    try:
        print(place_bid(bid_data, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Place a bid as non buyer
    print("\nTest: Place a bid as non buyer")
    bid_data.quantity = 5.0
    try:
        print(place_bid(bid_data, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Get bids with email only (buyer)
    print("\nTest: Get bids with email only (buyer)")
    try:
        bids = get_bids(email="buyer@example.com")
        print(bids)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get bids with email only (admin)
    print("\nTest: Get bids with email only (admin)")
    try:
        bids = get_bids(email="admin@example.com")
        print(bids)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get bids with email and listing_id (buyer)
    print("\nTest: Get bids with email and listing_id (buyer)")
    try:
        bids = get_bids(email="buyer@example.com", listing_id="listing123")
        print(bids)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get bids with email and listing_id (admin)
    print("\nTest: Get bids with email and listing_id (admin)")
    try:
        bids = get_bids(email="admin@example.com", listing_id="listing123")
        print(bids)
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Get bids with no parameters
    print("\nTest: Get bids with no parameters")
    try:
        bids = get_bids()
        print(bids)
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Update a bid as the buyer
    print("\nTest: Update a bid as the buyer")

    update_data = BidCreateModel(
        listing_id=crop_listing_id,
        quantity=15.0,
        price_per_kg=22.0,
    )
    try:
        print(update_bid(bid_id, update_data, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Update a bid as non buyer
    print("\nTest: Update a bid as non buyer")

    update_data = BidCreateModel(
        listing_id=crop_listing_id,
        quantity=15.0,
        price_per_kg=22.0,
    )
    try:
        print(update_bid(bid_id, update_data, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

        # Test case: Update a non existant bid
    print("\nTest: Update a non existant bid")

    update_data = BidCreateModel(
        listing_id=crop_listing_id,
        quantity=15.0,
        price_per_kg=22.0,
    )
    try:
        print(update_bid("asdasds", update_data, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Accept a bid non farmer
    print("\nTest: Accept a bid as the buyer")
    try:
        print(accept_bid(bid_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Accept a non existant bid
    print("\nTest: Accept a non existant bid")
    try:
        print(accept_bid("asdasdas", "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # Test case: Accept a bid as the farmer
    accepted_bid_id = None
    print("\nTest: Accept a bid as the farmer")
    try:
        accepted_bid_id = accept_bid(bid_id, "farmer@example.com")
        print(accepted_bid_id)
        accepted_bid_id = accepted_bid_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject a non existant bid
    print("\nTest: Reject a non existant bid")
    try:
        print(reject_bid("aasdasda", "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject a bid as non farmer
    print("\nTest: Reject a bid as non farmer")
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(reject_bid(bid_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Reject a bid as the farmer
    print("\nTest: Reject a bid as the farmer")
    try:
        print(reject_bid(bid_id, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    rejected_bid_id = bid_id

    # Test case: Delete a bid as non buyer
    print("\nTest: Delete a bid as non buyer")
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(delete_bid(bid_id, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a bid as the buyer
    print("\nTest: Delete a bid as the buyer")
    try:
        print(delete_bid(bid_id, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a bid as admin
    print("\nTest: Delete a bid as admin")
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(delete_bid(bid_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a rejected bid as the admin
    print("\nTest: Delete a rejected bid as the admin")
    try:
        print(delete_bid(rejected_bid_id, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a accepted bid as the buyer
    print("\nTest: Delete a accepted bid as the buyer")
    try:
        print(delete_bid(accepted_bid_id, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Delete a rejected bid as the admin
    print("\nTest: Delete a accepted bid as the admin")
    try:
        print(delete_bid(accepted_bid_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Fulfill a bid as the farmer
    print("\nTest: Fulfill a bid as the farmer")
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
        accept_bid(bid_id, "farmer@example.com")
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(fulfill_bid(bid_id, "farmer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Fulfill a bid as the buyer
    print("\nTest: Fulfill a bid as the buyer")
    try:
        print(fulfill_bid(bid_id, "buyer@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Test case: Fulfill a bid as admin
    print("\nTest: Fulfill a bid as admin")
    try:
        bid_id = place_bid(bid_data, "buyer@example.com")
        print(bid_id)
        bid_id = bid_id["$id"]
        accept_bid(bid_id, "farmer@example.com")
    except HTTPException as e:
        print(f"Error: {e.detail}")
    try:
        print(fulfill_bid(bid_id, "admin@example.com"))
    except HTTPException as e:
        print(f"Error: {e.detail}")
    # assuming same farmer who created is allowed to accept, reject and same buyer who placed bid can fulfill,update,delete etc
    # delete all bids and listings. for listing use id and for bids use time
    print("\nDelete all bids and listings")
    try:
        # delete all bids
        bids = DATABASES.list_documents(DATABASE_ID, COLLECTION_BIDS)
        for bid in bids["documents"]:
            DATABASES.delete_document(DATABASE_ID, COLLECTION_BIDS, bid["$id"])
        # delete all listings
        listings = DATABASES.list_documents(DATABASE_ID, COLLECTION_CROP_LISTINGS)
        for listing in listings["documents"]:
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_CROP_LISTINGS, listing["$id"]
            )
    except Exception as e:
        print(f"Error: {e.detail}")


def test_reject_bids_functions():
    # Step 1: Create a crop listing and store its ID

    try:
        listing_data = CropListingCreateModel(
            crop_type="Wheat",
            price_per_kg=20.0,
            total_quantity=10.0,
        )
        created_listing = create_crop_listing(listing_data, "farmer@example.com")
        listing_id = created_listing["$id"]  # Store the listing ID

    except HTTPException as e:
        print(f"Error creating listing: {e.detail}")
        return  # Exit the test if listing creation fails

    # Step 2: Place multiple bids for the created listing

    try:
        bid_data_1 = BidCreateModel(
            quantity=8.0,
            price_per_kg=20.0,
            listing_id=listing_id,
        )
        created_bid_1 = place_bid(bid_data_1, "buyer@example.com")
        bid_id_1 = created_bid_1["$id"]  # Store the first bid ID

        bid_data_2 = BidCreateModel(
            quantity=3.0,
            price_per_kg=20.0,
            listing_id=listing_id,
        )
        created_bid_2 = place_bid(bid_data_2, "buyer@example.com")
        bid_id_3 = place_bid(
            BidCreateModel(
                quantity=2.0,
                price_per_kg=20.0,
                listing_id=listing_id,
            ),
            "buyer@example.com",
        )["$id"]
        bid_id_4 = place_bid(
            BidCreateModel(
                quantity=2.0,
                price_per_kg=20.0,
                listing_id=listing_id,
            ),
            "buyer@example.com",
        )["$id"]
        bid_id_2 = created_bid_2["$id"]  # Store the second bid ID
    except HTTPException as e:
        print(f"Error placing bids: {e.detail}")
        return  # Exit the test if bid placement fails

    # Step 3: Accept a bid to reduce available quantity (triggers reject_bids_below_available_quantity)

    try:
        accept_bid(bid_id_1, "farmer@example.com")
        bid_2_status = DATABASES.get_document(DATABASE_ID, COLLECTION_BIDS, bid_id_2)[
            "status"
        ]
        print(f"Bid ID 2 status after accepting bid ID 1: {bid_2_status}")
    except HTTPException as e:
        print(f"Error: {e.detail}")

    try:
        accept_bid(bid_id_3, "farmer@example.com")
        bid_4_status = DATABASES.get_document(DATABASE_ID, COLLECTION_BIDS, bid_id_4)[
            "status"
        ]
        print(f"Bid ID 4 status after accepting bid ID 2: {bid_4_status}")
    except HTTPException as e:
        print(f"Error: {e.detail}")

    # Step 5: Cancel the listing (triggers reject_pending_bids)

    try:
        listing_data = CropListingCreateModel(
            crop_type="Wheat",
            price_per_kg=20.0,
            total_quantity=10.0,
        )
        created_listing = create_crop_listing(listing_data, "farmer@example.com")
        listing_id = created_listing["$id"]  # Store the listing ID

        bid_id_4 = place_bid(
            BidCreateModel(
                quantity=2.0,
                price_per_kg=20.0,
                listing_id=listing_id,
            ),
            "buyer@example.com",
        )["$id"]
    except HTTPException as e:
        print(f"Error creating listing: {e.detail}")
        return  # Exit the test if listing creation fails

    try:
        cancel_crop_listing(listing_id, "admin@example.com")
        bid_4_status = DATABASES.get_document(DATABASE_ID, COLLECTION_BIDS, bid_id_4)[
            "status"
        ]
        print(f"Bid ID 4 status after canceling listing: {bid_4_status}")
    except HTTPException as e:
        print(f"Error: {e.detail}")

    try:
        # delete all bids
        bids = DATABASES.list_documents(DATABASE_ID, COLLECTION_BIDS)
        for bid in bids["documents"]:
            DATABASES.delete_document(DATABASE_ID, COLLECTION_BIDS, bid["$id"])
        # delete all listings
        listings = DATABASES.list_documents(DATABASE_ID, COLLECTION_CROP_LISTINGS)
        for listing in listings["documents"]:
            DATABASES.delete_document(
                DATABASE_ID, COLLECTION_CROP_LISTINGS, listing["$id"]
            )
    except Exception as e:
        print(f"Error: {e.detail}")


def run_tests():
    test_listings()
    test_bids()
    test_reject_bids_functions()
