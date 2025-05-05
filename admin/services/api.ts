// API service for interacting with the backend

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'https://4f70-124-66-175-40.ngrok-free.app';

// User interface matching the API response structure
export interface User {
  $id: string;
  $createdAt: string;
  $updatedAt: string;
  $permissions: string[];
  $databaseId: string;
  $collectionId: string;
  name: string;
  email: string;
  role: string;
  address?: string;
  zipcode?: string; // Added zipcode field
}

// Subsidy interfaces
export interface Subsidy {
  $id: string;
  $createdAt: string;
  $updatedAt: string;
  program: string;
  description: string;
  eligibility: string;
  type: string;
  benefits: string;
  application_process: string;
  locations: string[];
  dynamic_fields: string;
  max_recipients: number;
  provider: string;
  status: string;
  submitted_by?: string;
  recipients_accepted?: number;
}

export interface SubsidyRequest {
  $id: string;
  $createdAt: string;
  $updatedAt: string;
  subsidy_id: string;
  farmer_id: string;
  status: string;
}

interface UsersListResponse {
  total: number;
  documents: User[];
}

interface SubsidiesListResponse {
  total: number;
  documents: Subsidy[];
}

interface SubsidyRequestsListResponse {
  total: number;
  documents: SubsidyRequest[];
}

interface ApiError {
  message: string;
  status?: number;
}

// Listing interfaces
export interface Listing {
  $id: string;
  $createdAt: string;
  $updatedAt: string;
  crop_type: string;
  price_per_kg: number;
  total_quantity: number;
  available_quantity: number;
  status: string;
  farmer_id: string;
}

export interface ListingsResponse {
  total: number;
  documents: Listing[];
}

// Bid interfaces
export interface Bid {
  $id: string;
  $createdAt: string;
  $updatedAt: string;
  quantity: number;
  price_per_kg: number;
  listing_id: string;
  buyer_id: string;
  status: string;
}

export interface BidsResponse {
  total: number;
  documents: Bid[];
}

// Helper function for API requests
async function apiRequest<T>(endpoint: string, options?: RequestInit): Promise<T> {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'ngrok-skip-browser-warning': 'true',
        'Content-Type': 'application/json',
        ...options?.headers,
      },
    });
    // Check if the response is ok (status 200-299)
    console.log(response);
    if (!response.ok) {
      const errorData = await response.json();
      console.error(`API error (${response.status}):`, errorData);
      throw {
        message: errorData.message || `API request failed with status ${response.status}`,
        status: response.status,
      };
    }

    return await response.json();
  } catch (error) {
    console.error('API request failed:', error);
    throw error instanceof Error ? error : new Error('Unknown API error');
  }
}

/**
 * Get a list of users, optionally filtered by role
 * @param role Optional role filter (admin, farmer, buyer, provider, or all)
 * @returns UsersListResponse containing total count and array of User objects
 */
export async function getUsers(role: string = 'all'): Promise<UsersListResponse> {
  console.debug('Fetching users with role filter:', role);
  return apiRequest<UsersListResponse>(`/users?type=${role}`);
}

/**
 * Get a specific user by email
 * @param email User's email
 * @returns User object
 */
export async function getUserByEmail(email: string): Promise<User> {
  console.debug('Fetching user by email:', email);
  return apiRequest<User>(`/users/${encodeURIComponent(email)}`);
}

/**
 * Create a new user
 * @param userData User data to create
 * @returns Created User object
 */
export async function createUser(userData: {
  name: string;
  email: string;
  role: string;
  address?: string;
  zipcode?: string; // Added zipcode field
}): Promise<User> {
  console.debug('Creating new user:', userData);
  return apiRequest<User>('/users', {
    method: 'POST',
    body: JSON.stringify(userData),
  });
}

/**
 * Update an existing user
 * @param email User's email
 * @param userData Updated user data
 * @returns Updated User object
 */
export async function updateUser(
  email: string,
  userData: {
    name?: string;
    email?: string;
    role?: string;
    address?: string;
    zipcode?: string; // Added zipcode field
  }
): Promise<User> {
  console.debug('Updating user:', email, userData);
  return apiRequest<User>(`/users/${encodeURIComponent(email)}`, {
    method: 'PATCH',
    body: JSON.stringify(userData),
  });
}

/**
 * Delete a user
 * @param email User's email to delete
 * @param requesterEmail Email of the user making the request (must be the user or an admin)
 * @returns Success message object { message: string }
 */
export async function deleteUser(email: string, requesterEmail: string): Promise<{ message: string }> {
  console.debug('Deleting user:', email, 'requested by:', requesterEmail);
  return apiRequest<{ message: string }>(
    `/users/${encodeURIComponent(email)}?requester_email=${encodeURIComponent(requesterEmail)}`,
    {
      method: 'DELETE',
    }
  );
}

/**
 * Get a user's role
 * @param email User's email
 * @returns Object containing user's role { role: string }
 */
export async function getUserRole(email: string): Promise<{ role: string }> {
  console.debug('Fetching role for user:', email);
  return apiRequest<{ role: string }>(`/users/${encodeURIComponent(email)}/role`);
}

/**
 * Get a list of subsidies
 * @param email User's email
 * @param type Optional filter by type (cash, asset, training, loan, or all)
 * @param status Optional filter by status (listed, removed, fulfilled, or all)
 * @param provider Optional filter by provider organization
 * @returns SubsidiesListResponse containing total count and array of Subsidy objects
 */
export async function getSubsidies(
  email?: string, 
  type?: string, 
  status?: string, 
  provider?: string
): Promise<SubsidiesListResponse> {
  console.debug('Fetching subsidies with filters:', { email, type, status, provider });
  let endpoint = '/subsidies'; // Changed back to '/subsidies' based on API documentation
  const params = new URLSearchParams();
  
  if (email) params.append('email', email);
  if (type) params.append('type', type);
  if (status) params.append('status', status);
  if (provider) params.append('provider', provider);
  
  const queryString = params.toString();
  if (queryString) endpoint += `?${queryString}`;
  
  return apiRequest<SubsidiesListResponse>(endpoint);
}

/**
 * Create a new subsidy (admin only)
 * @param subsidyData Object containing subsidy details
 * @param email Email of the admin creating the subsidy
 * @returns Created Subsidy object
 */
export async function createSubsidy(subsidyData: {
  program: string;
  description: string;
  eligibility: string;
  type: string;
  benefits: string;
  application_process: string;
  locations: string[];
  dynamic_fields: string;
  max_recipients: number;
  provider: string;
}, email: string): Promise<Subsidy> {
  console.debug('Creating subsidy:', subsidyData);
  let endpoint = '/subsidies';
  if (email) endpoint += `?email=${encodeURIComponent(email)}`;
  
  return apiRequest<Subsidy>(endpoint, {
    method: 'POST',
    body: JSON.stringify(subsidyData),
  });
}

/**
 * Update an existing subsidy (admin only)
 * @param subsidyId ID of the subsidy to update
 * @param email Email of the admin updating the subsidy
 * @param subsidyData Updated subsidy data
 * @returns Updated Subsidy object
 */
export async function updateSubsidy(
  subsidyId: string,
  email: string,
  subsidyData: Partial<{
    program: string;
    description: string;
    eligibility: string;
    type: string;
    benefits: string;
    application_process: string;
    locations: string[];
    dynamic_fields: string;
    max_recipients: number;
    provider: string;
  }>
): Promise<Subsidy> {
  console.debug('Updating subsidy:', subsidyId, subsidyData);
  let endpoint = `/subsidies/${subsidyId}`;
  if (email) endpoint += `?email=${encodeURIComponent(email)}`;
  
  return apiRequest<Subsidy>(endpoint, {
    method: 'PATCH',
    body: JSON.stringify(subsidyData),
  });
}

/**
 * Delete (remove) a subsidy (admin only)
 * @param subsidyId ID of the subsidy to delete
 * @param email Email of the admin deleting the subsidy
 * @returns Updated Subsidy object with status set to 'removed'
 */
export async function deleteSubsidy(subsidyId: string, email: string): Promise<Subsidy> {
  console.debug('Deleting subsidy:', subsidyId);
  let endpoint = `/subsidies/${subsidyId}`;
  if (email) endpoint += `?email=${encodeURIComponent(email)}`;
  
  return apiRequest<Subsidy>(endpoint, {
    method: 'DELETE',
  });
}

/**
 * Approve a subsidy
 * @param subsidyId ID of the subsidy to approve
 * @param email Email of the admin approving the subsidy
 * @returns Updated Subsidy object with status set to 'approved'
 */
export async function approveSubsidy(subsidyId: string, email: string): Promise<Subsidy> {
  console.debug('Approving subsidy:', subsidyId);
  return apiRequest<Subsidy>(`/subsidies/${subsidyId}/approve?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Reject a subsidy
 * @param subsidyId ID of the subsidy to reject
 * @param email Email of the admin rejecting the subsidy
 * @returns Updated Subsidy object with status set to 'rejected'
 */
export async function rejectSubsidy(subsidyId: string, email: string): Promise<Subsidy> {
  console.debug('Rejecting subsidy:', subsidyId);
  return apiRequest<Subsidy>(`/subsidies/${subsidyId}/reject?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Get subsidy requests
 * @param email User's email (required for farmers, optional for admins)
 * @param status Optional filter by status (requested, accepted, rejected, withdrawn, removed, fulfilled, or all)
 * @param subsidyId Optional filter by subsidy ID
 * @returns SubsidyRequestsListResponse containing total count and array of SubsidyRequest objects
 */
export async function getSubsidyRequests(
  email: string, 
  status?: string, 
  subsidyId?: string
): Promise<SubsidyRequestsListResponse> {
  console.debug('Fetching subsidy requests:', { email, status, subsidyId });
  let endpoint = `/subsidy_requests?email=${encodeURIComponent(email)}`;
  if (status) {
    endpoint += `&status=${encodeURIComponent(status)}`;
  }
  if (subsidyId) {
    endpoint += `&subsidy_id=${encodeURIComponent(subsidyId)}`;
  }
  return apiRequest<SubsidyRequestsListResponse>(endpoint);
}

/**
 * Create a subsidy request (farmers only)
 * @param email Email of the farmer requesting the subsidy
 * @param subsidyId ID of the subsidy being requested
 * @returns Created SubsidyRequest object with status set to 'requested'
 */
export async function createSubsidyRequest(
  email: string,
  subsidyId: string
): Promise<SubsidyRequest> {
  console.debug('Creating subsidy request for:', subsidyId);
  return apiRequest<SubsidyRequest>(`/subsidy_requests?email=${encodeURIComponent(email)}`, {
    method: 'POST',
    body: JSON.stringify({ subsidy_id: subsidyId }),
  });
}

/**
 * Accept a subsidy request (admin only)
 * @param requestId ID of the request to accept
 * @param email Email of the admin accepting the request
 * @returns Updated SubsidyRequest object with status set to 'accepted'
 */
export async function acceptSubsidyRequest(requestId: string, email: string): Promise<SubsidyRequest> {
  console.debug('Accepting subsidy request:', requestId);
  return apiRequest<SubsidyRequest>(`/subsidy_requests/${requestId}/accept?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Reject a subsidy request (admin only)
 * @param requestId ID of the request to reject
 * @param email Email of the admin rejecting the request
 * @returns Updated SubsidyRequest object with status set to 'rejected'
 */
export async function rejectSubsidyRequest(requestId: string, email: string): Promise<SubsidyRequest> {
  console.debug('Rejecting subsidy request:', requestId);
  return apiRequest<SubsidyRequest>(`/subsidy_requests/${requestId}/reject?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Delete (withdraw/remove) a subsidy request
 * @param requestId ID of the request to delete
 * @param email Email of the user deleting the request (farmer or admin)
 * @returns Updated SubsidyRequest object with status set to 'withdrawn' or 'removed'
 */
export async function deleteSubsidyRequest(requestId: string, email: string): Promise<SubsidyRequest> {
  console.debug('Deleting subsidy request:', requestId);
  return apiRequest<SubsidyRequest>(`/subsidy_requests/${requestId}?email=${encodeURIComponent(email)}`, {
    method: 'DELETE',
  });
}

/**
 * Fulfill a subsidy request
 * @param requestId ID of the request to fulfill
 * @param email Email of the user fulfilling the request (farmer or admin)
 * @returns Updated SubsidyRequest object with status set to 'fulfilled'
 */
export async function fulfillSubsidyRequest(requestId: string, email: string): Promise<SubsidyRequest> {
  console.debug('Fulfilling subsidy request:', requestId);
  return apiRequest<SubsidyRequest>(`/subsidy_requests/${requestId}/fulfill?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
    body: JSON.stringify({ email }),
  });
}

/**
 * Get all marketplace listings with optional filters
 * @param email Optional email to filter by farmer
 * @param type Optional filter by listing status
 * @returns ListingsResponse containing total count and array of Listing objects
 */
export async function getListings(email?: string, type: string = 'all'): Promise<ListingsResponse> {
  console.debug('Fetching listings with filters:', { email, type });
  let endpoint = `/listing`;
  const params = new URLSearchParams();
  if (email) params.append('email', email);
  if (type !== 'all') params.append('type', type);
  
  const queryString = params.toString();
  if (queryString) endpoint += `?${queryString}`;
  
  return apiRequest<ListingsResponse>(endpoint);
}

/**
 * Get a specific listing by ID
 * @param listingId ID of the listing
 * @returns Listing object with full listing details
 */
export async function getListingById(listingId: string): Promise<Listing> {
  console.debug('Fetching listing by ID:', listingId);
  return apiRequest<Listing>(`/listing/${listingId}`);
}

/**
 * Create a new listing (farmers only)
 * @param email Email of the farmer creating the listing
 * @param listingData Listing data
 * @returns Created Listing object with status set to 'listed'
 */
export async function createListing(
  email: string,
  listingData: {
    crop_type: string;
    price_per_kg: number;
    total_quantity: number;
  }
): Promise<Listing> {
  console.debug('Creating new listing:', listingData);
  return apiRequest<Listing>(`/listing?email=${encodeURIComponent(email)}`, {
    method: 'POST',
    body: JSON.stringify(listingData),
  });
}

/**
 * Update an existing listing
 * @param listingId ID of the listing to update
 * @param email Email of the user updating the listing
 * @param listingData Updated listing data
 * @returns Updated Listing object
 */
export async function updateListing(
  listingId: string,
  email: string,
  listingData: {
    crop_type?: string;
    price_per_kg?: number;
    total_quantity?: number;
  }
): Promise<Listing> {
  console.debug('Updating listing:', listingId, listingData);
  return apiRequest<Listing>(`/listing/${listingId}?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
    body: JSON.stringify(listingData),
  });
}

/**
 * Cancel or remove a listing
 * @param listingId ID of the listing to cancel
 * @param email Email of the user canceling the listing
 * @returns Updated Listing object with status set to 'cancelled' or 'removed'
 */
export async function cancelListing(listingId: string, email: string): Promise<Listing> {
  console.debug('Canceling listing:', listingId);
  return apiRequest<Listing>(`/listing/${listingId}?email=${encodeURIComponent(email)}`, {
    method: 'DELETE',
  });
}

/**
 * Get all bids with optional filters
 * @param email Email of the user fetching the bids
 * @param type Optional filter by bid status
 * @param listingId Optional filter by listing ID
 * @returns BidsResponse containing total count and array of Bid objects
 */
export async function getBids(email: string, type: string = 'all', listingId?: string): Promise<BidsResponse> {
  console.debug('Fetching bids with filters:', { email, type, listingId });
  let endpoint = `/bids?email=${encodeURIComponent(email)}`;
  if (type !== 'all') endpoint += `&type=${encodeURIComponent(type)}`;
  if (listingId) endpoint += `&listing_id=${encodeURIComponent(listingId)}`;
  
  return apiRequest<BidsResponse>(endpoint);
}

/**
 * Place a bid on a listing
 * @param email Email of the buyer placing the bid
 * @param bidData Bid data
 * @returns Created Bid object with status set to 'pending'
 */
export async function placeBid(
  email: string,
  bidData: {
    quantity: number;
    price_per_kg: number;
    listing_id: string;
  }
): Promise<Bid> {
  console.debug('Placing bid:', bidData);
  return apiRequest<Bid>(`/bids?email=${encodeURIComponent(email)}`, {
    method: 'POST',
    body: JSON.stringify(bidData),
  });
}

/**
 * Accept a bid
 * @param bidId ID of the bid to accept
 * @param email Email of the farmer accepting the bid
 * @returns Updated Bid object with status set to 'accepted'
 */
export async function acceptBid(bidId: string, email: string): Promise<Bid> {
  console.debug('Accepting bid:', bidId);
  return apiRequest<Bid>(`/bids/accept/${bidId}?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Reject a bid
 * @param bidId ID of the bid to reject
 * @param email Email of the farmer rejecting the bid
 * @returns Updated Bid object with status set to 'rejected'
 */
export async function rejectBid(bidId: string, email: string): Promise<Bid> {
  console.debug('Rejecting bid:', bidId);
  return apiRequest<Bid>(`/bids/reject/${bidId}?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

/**
 * Update a bid
 * @param bidId ID of the bid to update
 * @param email Email of the buyer updating the bid
 * @param bidData Updated bid data
 * @returns Updated Bid object
 */
export async function updateBid(
  bidId: string,
  email: string,
  bidData: {
    quantity?: number;
    price_per_kg?: number;
  }
): Promise<Bid> {
  console.debug('Updating bid:', bidId, bidData);
  return apiRequest<Bid>(`/bids/update/${bidId}?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
    body: JSON.stringify(bidData),
  });
}

/**
 * Delete a bid
 * @param bidId ID of the bid to delete
 * @param email Email of the user deleting the bid
 * @returns Success message object { message: string }
 */
export async function deleteBid(bidId: string, email: string): Promise<{ message: string }> {
  console.debug('Deleting bid:', bidId);
  return apiRequest<{ message: string }>(`/bids/${bidId}?email=${encodeURIComponent(email)}`, {
    method: 'DELETE',
  });
}

/**
 * Fulfill a bid
 * @param bidId ID of the bid to fulfill
 * @param email Email of the user fulfilling the bid
 * @returns Updated Bid object with status set to 'fulfilled'
 */
export async function fulfillBid(bidId: string, email: string): Promise<Bid> {
  console.debug('Fulfilling bid:', bidId);
  return apiRequest<Bid>(`/bids/fulfill/${bidId}?email=${encodeURIComponent(email)}`, {
    method: 'PATCH',
  });
}

// Mock API for development/testing if needed
export const mockApi = {
  users: [
    {
      $id: '1',
      $createdAt: '2023-01-01T00:00:00.000Z',
      $updatedAt: '2023-01-01T00:00:00.000Z',
      $permissions: [],
      $databaseId: 'agri_marketplace',
      $collectionId: 'users',
      name: 'Admin User',
      email: 'admin@example.com',
      role: 'admin',
      address: '123 Admin Street',
      zipcode: '10001' // Added zipcode field
    },
    {
      $id: '2',
      $createdAt: '2023-01-02T00:00:00.000Z',
      $updatedAt: '2023-01-02T00:00:00.000Z',
      $permissions: [],
      $databaseId: 'agri_marketplace',
      $collectionId: 'users',
      name: 'Farmer Joe',
      email: 'farmer@example.com',
      role: 'farmer',
      address: '456 Farm Road',
      zipcode: '20002' // Added zipcode field
    },
    {
      $id: '3',
      $createdAt: '2023-01-03T00:00:00.000Z',
      $updatedAt: '2023-01-03T00:00:00.000Z',
      $permissions: [],
      $databaseId: 'agri_marketplace',
      $collectionId: 'users',
      name: 'Buyer Bob',
      email: 'buyer@example.com',
      role: 'buyer',
      address: '789 Market Avenue',
      zipcode: '30003' // Added zipcode field
    }
  ]
};
