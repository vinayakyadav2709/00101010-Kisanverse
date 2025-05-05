import React, { useState, useEffect } from 'react';
import { 
  View, 
  StyleSheet, 
  SafeAreaView, 
  ScrollView, 
  TouchableOpacity, 
  FlatList,
  ActivityIndicator,
  Alert,
  TextInput,
  Modal
} from 'react-native';
import { useTheme } from '../../context/ThemeContext';
import { Typography } from '../../components/Typography';
import { Card } from '../../components/Card';
import { useUserStore } from '../../store/userStore';
import { 
  ShoppingBag, 
  Package, 
  DollarSign, 
  ArrowRight, 
  AlertCircle,
  Calendar,
  Tag,
  Edit,
  X,
  Check
} from 'react-native-feather';

const API_BASE_URL = 'https://dcf2-124-66-175-40.ngrok-free.app';

// Types
type CropListing = {
  $id: string;
  crop_type: string;
  price_per_kg: number;
  total_quantity: number;
  available_quantity: number;
  status: string;
  farmer_id: string;
  $createdAt: string;
  $updatedAt: string;
};

type Bid = {
  $id: string;
  quantity: number;
  price_per_kg: number;
  listing_id: string;
  buyer_id: string;
  status: string;
  $createdAt: string;
  $updatedAt: string;
};

export default function MarketplaceScreen() {
  const { colors, spacing, radius } = useTheme();
  const { user } = useUserStore();
  
  // State
  const [activeTab, setActiveTab] = useState('listings');
  const [listings, setListings] = useState<CropListing[]>([]);
  const [myBids, setMyBids] = useState<Bid[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  
  // Bidding modal state
  const [bidModalVisible, setBidModalVisible] = useState(false);
  const [selectedListing, setSelectedListing] = useState<CropListing | null>(null);
  const [bidQuantity, setBidQuantity] = useState('');
  const [bidPrice, setBidPrice] = useState('');
  
  // Edit bid modal state
  const [editBidModalVisible, setEditBidModalVisible] = useState(false);
  const [selectedBid, setSelectedBid] = useState<Bid | null>(null);
  const [editBidQuantity, setEditBidQuantity] = useState('');
  const [editBidPrice, setEditBidPrice] = useState('');

  // Fetch listings and bids
  useEffect(() => {
    fetchListingsAndBids();
  }, [user]);

  const fetchListingsAndBids = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!user?.email) {
        throw new Error('User email not found');
      }

      // Include zip code in the query if available
      const zipCodeParam = user?.zipcode ? `&zipCode=${user.zipcode}` : '';
      
      // Fetch crop listings with zip code preference when available
      const listingsResponse = await fetch(`${API_BASE_URL}/listing?type=listed${zipCodeParam}`);
      if (!listingsResponse.ok) {
        throw new Error(`Failed to fetch listings: ${listingsResponse.status}`);
      }
      const listingsData = await listingsResponse.json();
      
      // Fetch user's bids
      const bidsResponse = await fetch(`${API_BASE_URL}/bids?email=${user.email}`);
      if (!bidsResponse.ok) {
        throw new Error(`Failed to fetch bids: ${bidsResponse.status}`);
      }
      const bidsData = await bidsResponse.json();
      
      setListings(listingsData.documents || []);
      setMyBids(bidsData.documents || []);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching data:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      setLoading(false);
      
      // Use mock data as fallback
      setListings([
        {
          $id: "1",
          crop_type: "Wheat",
          price_per_kg: 20.0,
          total_quantity: 100.0,
          available_quantity: 80.0,
          status: "listed",
          farmer_id: "farmer_id",
          $createdAt: "2025-04-26T12:30:03.816+00:00",
          $updatedAt: "2025-04-26T12:30:03.816+00:00"
        },
        {
          $id: "2",
          crop_type: "Rice",
          price_per_kg: 35.0,
          total_quantity: 200.0,
          available_quantity: 200.0,
          status: "listed",
          farmer_id: "farmer_id",
          $createdAt: "2025-04-27T10:30:03.816+00:00",
          $updatedAt: "2025-04-27T10:30:03.816+00:00"
        }
      ]);
      
      setMyBids([
        {
          $id: "bid1",
          quantity: 10.0,
          price_per_kg: 18.0,
          listing_id: "1",
          buyer_id: user?.$id || "buyer_id",
          status: "pending",
          $createdAt: "2025-04-26T14:30:03.816+00:00",
          $updatedAt: "2025-04-26T14:30:03.816+00:00"
        }
      ]);
    }
  };

  // Place a bid
  const handlePlaceBid = async () => {
    if (!selectedListing) return;
    
    try {
      if (!bidQuantity || !bidPrice) {
        Alert.alert('Validation Error', 'Please enter both quantity and price');
        return;
      }
      
      const quantity = parseFloat(bidQuantity);
      const pricePerKg = parseFloat(bidPrice);
      
      if (isNaN(quantity) || quantity <= 0) {
        Alert.alert('Validation Error', 'Please enter a valid quantity');
        return;
      }
      
      if (isNaN(pricePerKg) || pricePerKg <= 0) {
        Alert.alert('Validation Error', 'Please enter a valid price');
        return;
      }
      
      if (quantity > selectedListing.available_quantity) {
        Alert.alert('Validation Error', `Available quantity is only ${selectedListing.available_quantity}kg`);
        return;
      }
      
      const response = await fetch(`${API_BASE_URL}/bids?email=${user?.email}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          quantity,
          price_per_kg: pricePerKg,
          listing_id: selectedListing.$id
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to place bid');
      }
      
      const newBid = await response.json();
      setMyBids([...myBids, newBid]);
      Alert.alert('Success', 'Your bid has been placed successfully');
      setBidModalVisible(false);
      setBidQuantity('');
      setBidPrice('');
      
      // Refresh data
      fetchListingsAndBids();
    } catch (error) {
      console.error('Error placing bid:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to place bid');
    }
  };

  // Update a bid
  const handleUpdateBid = async () => {
    if (!selectedBid) return;
    
    try {
      const quantity = parseFloat(editBidQuantity);
      const pricePerKg = parseFloat(editBidPrice);
      
      if (editBidQuantity && (isNaN(quantity) || quantity <= 0)) {
        Alert.alert('Validation Error', 'Please enter a valid quantity');
        return;
      }
      
      if (editBidPrice && (isNaN(pricePerKg) || pricePerKg <= 0)) {
        Alert.alert('Validation Error', 'Please enter a valid price');
        return;
      }
      
      const updateData: any = {};
      if (editBidQuantity) updateData.quantity = quantity;
      if (editBidPrice) updateData.price_per_kg = pricePerKg;
      
      const response = await fetch(`${API_BASE_URL}/bids/${selectedBid.$id}?email=${user?.email}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(updateData)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update bid');
      }
      
      const updatedBid = await response.json();
      setMyBids(myBids.map(bid => bid.$id === updatedBid.$id ? updatedBid : bid));
      Alert.alert('Success', 'Your bid has been updated successfully');
      setEditBidModalVisible(false);
      setEditBidQuantity('');
      setEditBidPrice('');
      
      // Refresh data
      fetchListingsAndBids();
    } catch (error) {
      console.error('Error updating bid:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to update bid');
    }
  };

  // Delete a bid
  const handleDeleteBid = async (bidId: string) => {
    try {
      Alert.alert(
        'Confirm Deletion',
        'Are you sure you want to withdraw this bid?',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Delete',
            style: 'destructive',
            onPress: async () => {
              const response = await fetch(`${API_BASE_URL}/bids/${bidId}?email=${user?.email}`, {
                method: 'DELETE'
              });
              
              if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to delete bid');
              }
              
              setMyBids(myBids.filter(bid => bid.$id !== bidId));
              Alert.alert('Success', 'Bid withdrawn successfully');
              
              // Refresh data
              fetchListingsAndBids();
            }
          }
        ]
      );
    } catch (error) {
      console.error('Error deleting bid:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to withdraw bid');
    }
  };

  // Mark bid as fulfilled (for testing)
  const handleFulfillBid = async (bidId: string) => {
    try {
      Alert.alert(
        'Confirm Fulfillment',
        'Are you sure you want to mark this bid as fulfilled?',
        [
          { text: 'Cancel', style: 'cancel' },
          {
            text: 'Mark Fulfilled',
            onPress: async () => {
              const response = await fetch(`${API_BASE_URL}/bids/${bidId}/fulfill?email=${user?.email}`, {
                method: 'PATCH'
              });
              
              if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Failed to fulfill bid');
              }
              
              // Update local state
              setMyBids(myBids.map(bid => 
                bid.$id === bidId ? { ...bid, status: 'fulfilled' } : bid
              ));
              
              Alert.alert('Success', 'Bid marked as fulfilled');
              
              // Refresh data
              fetchListingsAndBids();
            }
          }
        ]
      );
    } catch (error) {
      console.error('Error fulfilling bid:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to mark bid as fulfilled');
    }
  };

  // Open bid modal with pre-filled listing data
  const openBidModal = (listing: CropListing) => {
    setSelectedListing(listing);
    // Pre-fill bid price with listing price as default
    setBidPrice(listing.price_per_kg.toString());
    setBidModalVisible(true);
  };

  // Open edit bid modal with pre-filled bid data
  const openEditBidModal = (bid: Bid) => {
    setSelectedBid(bid);
    setEditBidQuantity(bid.quantity.toString());
    setEditBidPrice(bid.price_per_kg.toString());
    setEditBidModalVisible(true);
  };

  // Get the formatted date
  const formatDate = (dateString: string) => {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', { 
      year: 'numeric', 
      month: 'short', 
      day: 'numeric' 
    });
  };

  // Get color based on bid status
  const getBidStatusColor = (status: string) => {
    switch (status) {
      case 'accepted': return colors.success;
      case 'pending': return colors.warning;
      case 'rejected': return colors.error;
      case 'fulfilled': return colors.primary;
      default: return colors.textSecondary;
    }
  };

  // Render a crop listing item
  const renderListingItem = ({ item }: { item: CropListing }) => (
    <Card variant="elevated" style={styles.listingCard}>
      <View style={styles.cropTypeContainer}>
        <View style={[styles.cropIconContainer, { backgroundColor: colors.primary + '15' }]}>
          <Package width={20} height={20} stroke={colors.primary} />
        </View>
        <Typography variant="bodyLarge" style={styles.cropType}>
          {item.crop_type}
        </Typography>
      </View>
      
      <View style={styles.listingDetails}>
        <View style={styles.detailRow}>
          <Typography variant="body" color="textSecondary">Price:</Typography>
          <Typography variant="bodyLarge" style={styles.price}>
            ₹{item.price_per_kg}/kg
          </Typography>
        </View>
        
        <View style={styles.detailRow}>
          <Typography variant="body" color="textSecondary">Available:</Typography>
          <Typography variant="body">
            {item.available_quantity}kg of {item.total_quantity}kg
          </Typography>
        </View>
        
        <View style={styles.detailRow}>
          <Typography variant="body" color="textSecondary">Listed on:</Typography>
          <Typography variant="body">
            {formatDate(item.$createdAt)}
          </Typography>
        </View>
      </View>
      
      <TouchableOpacity
        style={[styles.bidButton, { backgroundColor: colors.primary }]}
        onPress={() => openBidModal(item)}
      >
        <Typography variant="body" style={{ color: 'white', fontWeight: '600' }}>
          Place Bid
        </Typography>
        <ArrowRight width={16} height={16} stroke="white" style={{ marginLeft: 8 }} />
      </TouchableOpacity>
    </Card>
  );

  // Render a bid item
  const renderBidItem = ({ item }: { item: Bid }) => {
    // Find the corresponding listing
    const listing = listings.find(l => l.$id === item.listing_id);
    
    return (
      <Card variant="elevated" style={styles.bidCard}>
        <View style={styles.bidHeader}>
          <View style={styles.bidCropInfo}>
            <View style={[styles.bidIconContainer, { backgroundColor: colors.primary + '15' }]}>
              <DollarSign width={18} height={18} stroke={colors.primary} />
            </View>
            <Typography variant="bodyLarge" style={styles.bidCropType}>
              {listing ? listing.crop_type : 'Unknown Crop'}
            </Typography>
          </View>
          
          <View style={[
            styles.bidStatusBadge, 
            { backgroundColor: getBidStatusColor(item.status) + '20' }
          ]}>
            <Typography 
              variant="small" 
              style={{ color: getBidStatusColor(item.status), fontWeight: '600' }}
            >
              {item.status.toUpperCase()}
            </Typography>
          </View>
        </View>
        
        <View style={styles.bidDetails}>
          <View style={styles.bidDetailRow}>
            <Calendar width={16} height={16} stroke={colors.textSecondary} />
            <Typography variant="caption" color="textSecondary" style={styles.bidDetailLabel}>
              Placed on:
            </Typography>
            <Typography variant="body">
              {formatDate(item.$createdAt)}
            </Typography>
          </View>
          
          <View style={styles.bidDetailRow}>
            <Tag width={16} height={16} stroke={colors.textSecondary} />
            <Typography variant="caption" color="textSecondary" style={styles.bidDetailLabel}>
              Quantity:
            </Typography>
            <Typography variant="body">
              {item.quantity}kg
            </Typography>
          </View>
          
          <View style={styles.bidDetailRow}>
            <DollarSign width={16} height={16} stroke={colors.textSecondary} />
            <Typography variant="caption" color="textSecondary" style={styles.bidDetailLabel}>
              Your bid:
            </Typography>
            <Typography variant="bodyLarge" style={{ fontWeight: '600' }}>
              ₹{item.price_per_kg}/kg
            </Typography>
          </View>
        </View>
        
        {item.status === 'pending' && (
          <View style={styles.bidActions}>
            <TouchableOpacity 
              style={[styles.bidActionButton, { backgroundColor: colors.primary + '15' }]}
              onPress={() => openEditBidModal(item)}
            >
              <Edit width={15} height={15} stroke={colors.primary} />
              <Typography variant="small" color="primary" style={{ marginLeft: 4, fontWeight: '500' }}>
                Edit
              </Typography>
            </TouchableOpacity>
            
            <TouchableOpacity 
              style={[styles.bidActionButton, { backgroundColor: colors.error + '15' }]}
              onPress={() => handleDeleteBid(item.$id)}
            >
              <X width={15} height={15} stroke={colors.error} />
              <Typography variant="small" style={{ color: colors.error, marginLeft: 4, fontWeight: '500' }}>
                Withdraw
              </Typography>
            </TouchableOpacity>
          </View>
        )}
        
        {item.status === 'accepted' && (
          <View style={styles.bidActions}>
            <TouchableOpacity 
              style={[styles.bidActionButton, { backgroundColor: colors.success + '15' }]}
              onPress={() => handleFulfillBid(item.$id)}
            >
              <Check width={15} height={15} stroke={colors.success} />
              <Typography variant="small" style={{ color: colors.success, marginLeft: 4, fontWeight: '500' }}>
                Mark Received
              </Typography>
            </TouchableOpacity>
          </View>
        )}
      </Card>
    );
  };

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.header}>
        <Typography variant="heading" style={styles.title}>Marketplace</Typography>
        <TouchableOpacity 
          style={[styles.refreshButton, { backgroundColor: colors.primary + '15' }]}
          onPress={fetchListingsAndBids}
        >
          <Typography variant="small" color="primary">Refresh</Typography>
        </TouchableOpacity>
      </View>
      
      <View style={styles.tabs}>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === 'listings' && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab('listings')}
        >
          <Typography 
            variant="body" 
            color={activeTab === 'listings' ? 'primary' : 'textSecondary'}
            style={{ fontWeight: activeTab === 'listings' ? '600' : '400' }}
          >
            Crop Listings
          </Typography>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === 'bids' && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab('bids')}
        >
          <Typography 
            variant="body" 
            color={activeTab === 'bids' ? 'primary' : 'textSecondary'}
            style={{ fontWeight: activeTab === 'bids' ? '600' : '400' }}
          >
            My Bids
          </Typography>
        </TouchableOpacity>
      </View>
      
      {loading ? (
        <View style={styles.loadingContainer}>
          <ActivityIndicator size="large" color={colors.primary} />
          <Typography variant="body" style={styles.loadingText}>Loading...</Typography>
        </View>
      ) : error ? (
        <View style={styles.errorContainer}>
          <AlertCircle width={32} height={32} stroke={colors.error} />
          <Typography variant="body" color="error" style={styles.errorText}>
            {error}
          </Typography>
          <Typography variant="caption" color="textSecondary">
            Showing mock data instead
          </Typography>
        </View>
      ) : (
        activeTab === 'listings' ? (
          <FlatList
            data={listings}
            renderItem={renderListingItem}
            keyExtractor={(item) => item.$id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <AlertCircle width={32} height={32} stroke={colors.textSecondary} />
                <Typography variant="body" color="textSecondary" style={styles.emptyText}>
                  No crop listings available
                </Typography>
              </View>
            }
          />
        ) : (
          <FlatList
            data={myBids}
            renderItem={renderBidItem}
            keyExtractor={(item) => item.$id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <AlertCircle width={32} height={32} stroke={colors.textSecondary} />
                <Typography variant="body" color="textSecondary" style={styles.emptyText}>
                  You haven't placed any bids yet
                </Typography>
              </View>
            }
          />
        )
      )}
      
      {/* Place Bid Modal */}
      <Modal
        visible={bidModalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => {
          setBidModalVisible(false);
          setBidQuantity('');
          setBidPrice('');
        }}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.card }]}>
            <Typography variant="subheading" style={styles.modalTitle}>
              Place a Bid
            </Typography>
            
            {selectedListing && (
              <View style={styles.selectedListingInfo}>
                <Typography variant="bodyLarge" style={{ fontWeight: '600' }}>
                  {selectedListing.crop_type}
                </Typography>
                <Typography variant="body" color="textSecondary">
                  Market Price: ₹{selectedListing.price_per_kg}/kg
                </Typography>
                <Typography variant="body" color="textSecondary">
                  Available: {selectedListing.available_quantity}kg
                </Typography>
              </View>
            )}
            
            <View style={styles.inputGroup}>
              <Typography variant="body" style={styles.inputLabel}>
                Quantity (kg)
              </Typography>
              <TextInput
                style={[styles.input, { 
                  backgroundColor: colors.background, 
                  color: colors.text,
                  borderColor: colors.border
                }]}
                value={bidQuantity}
                onChangeText={setBidQuantity}
                placeholder="Enter quantity in kg"
                placeholderTextColor={colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
            
            <View style={styles.inputGroup}>
              <Typography variant="body" style={styles.inputLabel}>
                Your Bid Price (₹/kg)
              </Typography>
              <TextInput
                style={[styles.input, { 
                  backgroundColor: colors.background, 
                  color: colors.text,
                  borderColor: colors.border
                }]}
                value={bidPrice}
                onChangeText={setBidPrice}
                placeholder="Enter your bid price per kg"
                placeholderTextColor={colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
            
            <View style={styles.modalActions}>
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: colors.backgroundSecondary }]}
                onPress={() => {
                  setBidModalVisible(false);
                  setBidQuantity('');
                  setBidPrice('');
                }}
              >
                <Typography variant="body" color="textSecondary">Cancel</Typography>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: colors.primary }]}
                onPress={handlePlaceBid}
              >
                <Typography variant="body" style={{ color: 'white' }}>Submit Bid</Typography>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
      
      {/* Edit Bid Modal */}
      <Modal
        visible={editBidModalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => {
          setEditBidModalVisible(false);
          setEditBidQuantity('');
          setEditBidPrice('');
        }}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.card }]}>
            <Typography variant="subheading" style={styles.modalTitle}>
              Edit Your Bid
            </Typography>
            
            <View style={styles.inputGroup}>
              <Typography variant="body" style={styles.inputLabel}>
                New Quantity (kg)
              </Typography>
              <TextInput
                style={[styles.input, { 
                  backgroundColor: colors.background, 
                  color: colors.text,
                  borderColor: colors.border
                }]}
                value={editBidQuantity}
                onChangeText={setEditBidQuantity}
                placeholder="Enter new quantity in kg"
                placeholderTextColor={colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
            
            <View style={styles.inputGroup}>
              <Typography variant="body" style={styles.inputLabel}>
                New Bid Price (₹/kg)
              </Typography>
              <TextInput
                style={[styles.input, { 
                  backgroundColor: colors.background, 
                  color: colors.text,
                  borderColor: colors.border
                }]}
                value={editBidPrice}
                onChangeText={setEditBidPrice}
                placeholder="Enter your new bid price per kg"
                placeholderTextColor={colors.textSecondary}
                keyboardType="numeric"
              />
            </View>
            
            <View style={styles.modalActions}>
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: colors.backgroundSecondary }]}
                onPress={() => {
                  setEditBidModalVisible(false);
                  setEditBidQuantity('');
                  setEditBidPrice('');
                }}
              >
                <Typography variant="body" color="textSecondary">Cancel</Typography>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[styles.modalButton, { backgroundColor: colors.primary }]}
                onPress={handleUpdateBid}
              >
                <Typography variant="body" style={{ color: 'white' }}>Update Bid</Typography>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: '700',
  },
  refreshButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  tabs: {
    flexDirection: 'row',
    paddingHorizontal: 20,
    marginTop: 20,
    marginBottom: 16,
  },
  tab: {
    paddingVertical: 8,
    marginRight: 24,
  },
  activeTab: {
    borderBottomWidth: 3,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    marginTop: 16,
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    marginVertical: 16,
    textAlign: 'center',
  },
  listContainer: {
    padding: 20,
    paddingTop: 0,
    paddingBottom: 80,
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 60,
    padding: 20,
  },
  emptyText: {
    marginTop: 16,
    textAlign: 'center',
  },
  listingCard: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
  },
  cropTypeContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 16,
  },
  cropIconContainer: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 12,
  },
  cropType: {
    fontWeight: '700',
    fontSize: 18,
  },
  listingDetails: {
    marginBottom: 16,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  price: {
    fontWeight: '700',
  },
  bidButton: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
  bidCard: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
  },
  bidHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  bidCropInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  bidIconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  bidCropType: {
    fontWeight: '600',
  },
  bidStatusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  bidDetails: {
    marginBottom: 16,
  },
  bidDetailRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  bidDetailLabel: {
    marginLeft: 8,
    marginRight: 8,
  },
  bidActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  bidActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16,
    marginLeft: 8,
  },
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'center',
    alignItems: 'center',
  },
  modalContent: {
    width: '90%',
    padding: 20,
    borderRadius: 12,
  },
  modalTitle: {
    marginBottom: 16,
    fontWeight: '700',
  },
  selectedListingInfo: {
    marginBottom: 20,
    padding: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.05)',
    borderRadius: 8,
  },
  inputGroup: {
    marginBottom: 16,
  },
  inputLabel: {
    marginBottom: 8,
    fontWeight: '500',
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
  },
  modalActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 16,
  },
  modalButton: {
    flex: 1,
    paddingVertical: 12,
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 8,
  },
});
