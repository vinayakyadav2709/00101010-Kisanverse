import React, { useState, useEffect } from 'react'
import { View, Text, StyleSheet, FlatList, TextInput, TouchableOpacity, Alert } from 'react-native'
import { useTheme } from '../../context/ThemeContext'
import { Typography } from '../Typography'
import { Card } from '../Card'
import { Plus, DollarSign, Tag, Edit, X } from 'react-native-feather'

type Listing = {
  $id: string
  crop_type: string
  price_per_kg: number
  total_quantity: number
  available_quantity: number
  status: string
  farmer_id: string
  $createdAt: string
  $updatedAt: string
}

type Bid = {
  $id: string
  quantity: number
  price_per_kg: number
  listing_id: string
  buyer_id: string
  status: string
  $createdAt: string
}

const API_BASE_URL = 'https://4f70-124-66-175-40.ngrok-free.app';
const FARMER_EMAIL = 'farmer@example.com'; // This should come from auth context in a real app

const MarketListings = () => {
  const { colors, spacing, radius } = useTheme()
  const [listings, setListings] = useState<Listing[]>([])
  const [bids, setBids] = useState<{[key: string]: Bid[]}>({})
  const [loading, setLoading] = useState(true)
  const [showAddForm, setShowAddForm] = useState(false)
  const [formData, setFormData] = useState({
    crop_type: '',
    price_per_kg: '',
    total_quantity: ''
  })

  // Fetch farmer's listings
  useEffect(() => {
    const fetchListings = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/listing?email=${FARMER_EMAIL}&type=listed`)
        const data = await response.json()
        
        if (data && data.documents) {
          setListings(data.documents)
          
          // Fetch bids for each listing
          data.documents.forEach((listing: Listing) => {
            fetchBidsForListing(listing.$id)
          })
        }
        
        setLoading(false)
      } catch (error) {
        console.error('Error fetching listings:', error)
        setLoading(false)
        Alert.alert('Error', 'Failed to fetch listings. Please try again later.')
      }
    }
    
    fetchListings()
  }, [])
  
  const fetchBidsForListing = async (listingId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/bids?email=${FARMER_EMAIL}&listing_id=${listingId}`)
      const data = await response.json()
      
      if (data && data.documents) {
        setBids(prev => ({
          ...prev,
          [listingId]: data.documents
        }))
      }
    } catch (error) {
      console.error(`Error fetching bids for listing ${listingId}:`, error)
    }
  }
  
  const handleCreateListing = async () => {
    try {
      // Validate form inputs
      if (!formData.crop_type || !formData.price_per_kg || !formData.total_quantity) {
        Alert.alert('Error', 'All fields are required')
        return
      }
      
      const response = await fetch(`${API_BASE_URL}/listing?email=${FARMER_EMAIL}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          crop_type: formData.crop_type,
          price_per_kg: parseFloat(formData.price_per_kg),
          total_quantity: parseFloat(formData.total_quantity)
        })
      })
      
      const data = await response.json()
      
      if (response.ok) {
        // Add the new listing to the state
        setListings([...listings, data])
        setFormData({ crop_type: '', price_per_kg: '', total_quantity: '' })
        setShowAddForm(false)
        
        Alert.alert('Success', 'Listing created successfully')
      } else {
        Alert.alert('Error', data.message || 'Failed to create listing')
      }
    } catch (error) {
      console.error('Error creating listing:', error)
      Alert.alert('Error', 'Failed to create listing. Please try again later.')
    }
  }
  
  const handleCancelListing = async (listingId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/listing/${listingId}?email=${FARMER_EMAIL}`, {
        method: 'DELETE'
      })
      
      const data = await response.json()
      
      if (response.ok) {
        // Update local state - it will now be marked as cancelled
        setListings(listings.map(listing => 
          listing.$id === listingId ? { ...listing, status: 'cancelled' } : listing
        ))
        
        Alert.alert('Success', 'Listing cancelled successfully')
      } else {
        Alert.alert('Error', data.message || 'Failed to cancel listing')
      }
    } catch (error) {
      console.error('Error cancelling listing:', error)
      Alert.alert('Error', 'Failed to cancel listing. Please try again later.')
    }
  }
  
  const handleAcceptBid = async (listingId: string, bidId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/bids/${bidId}/accept?email=${FARMER_EMAIL}`, {
        method: 'PATCH'
      })
      
      const data = await response.json()
      
      if (response.ok) {
        // Update local state with the accepted bid
        setBids(prev => ({
          ...prev,
          [listingId]: prev[listingId].map(bid => 
            bid.$id === bidId ? { ...bid, status: 'accepted' } : bid
          )
        }))
        
        Alert.alert('Success', 'Bid accepted successfully')
      } else {
        Alert.alert('Error', data.message || 'Failed to accept bid')
      }
    } catch (error) {
      console.error('Error accepting bid:', error)
      Alert.alert('Error', 'Failed to accept bid. Please try again later.')
    }
  }
  
  const handleRejectBid = async (listingId: string, bidId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/bids/${bidId}/reject?email=${FARMER_EMAIL}`, {
        method: 'PATCH'
      })
      
      const data = await response.json()
      
      if (response.ok) {
        // Update local state with the rejected bid
        setBids(prev => ({
          ...prev,
          [listingId]: prev[listingId].map(bid => 
            bid.$id === bidId ? { ...bid, status: 'rejected' } : bid
          )
        }))
        
        Alert.alert('Success', 'Bid rejected successfully')
      } else {
        Alert.alert('Error', data.message || 'Failed to reject bid')
      }
    } catch (error) {
      console.error('Error rejecting bid:', error)
      Alert.alert('Error', 'Failed to reject bid. Please try again later.')
    }
  }
  
  const renderBid = (bid: Bid, listingId: string) => (
    <View key={bid.$id} style={[styles.bidItem, { borderColor: colors.border }]}>
      <View>
        <Typography variant="body">
          {bid.quantity}kg at ₹{bid.price_per_kg}/kg
        </Typography>
        <Typography variant="small" color="textSecondary">
          Total: ₹{bid.quantity * bid.price_per_kg}
        </Typography>
        <Typography variant="caption" color="textSecondary">
          Status: {bid.status}
        </Typography>
      </View>
      
      {bid.status === 'pending' && (
        <View style={styles.bidActions}>
          <TouchableOpacity 
            style={[styles.bidActionButton, { backgroundColor: colors.primary }]}
            onPress={() => handleAcceptBid(listingId, bid.$id)}
          >
            <Typography variant="small" color="accent">Accept</Typography>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[styles.bidActionButton, { backgroundColor: colors.error }]}
            onPress={() => handleRejectBid(listingId, bid.$id)}
          >
            <Typography variant="small" color="accent">Reject</Typography>
          </TouchableOpacity>
        </View>
      )}
    </View>
  )
  
  const renderListing = ({ item }: { item: Listing }) => (
    <Card variant="elevated" style={styles.listingCard}>
      <View style={styles.listingHeader}>
        <View style={styles.listingTitleSection}>
          <Typography variant="bodyLarge">{item.crop_type}</Typography>
          <View style={[styles.statusBadge, { 
            backgroundColor: item.status === 'listed' ? colors.success + '20' : colors.error + '20',
          }]}>
            <Typography variant="small" color={item.status === 'listed' ? 'success' : 'error'}>
              {item.status}
            </Typography>
          </View>
        </View>
        
        {item.status === 'listed' && (
          <TouchableOpacity 
            style={[styles.cancelButton, { borderColor: colors.error }]}
            onPress={() => handleCancelListing(item.$id)}
          >
            <X width={16} height={16} stroke={colors.error} />
          </TouchableOpacity>
        )}
      </View>
      
      <View style={styles.listingDetails}>
        <View style={styles.detailRow}>
          <Typography variant="body">Price:</Typography>
          <Typography variant="body">₹{item.price_per_kg}/kg</Typography>
        </View>
        <View style={styles.detailRow}>
          <Typography variant="body">Total Quantity:</Typography>
          <Typography variant="body">{item.total_quantity}kg</Typography>
        </View>
        <View style={styles.detailRow}>
          <Typography variant="body">Available:</Typography>
          <Typography variant="body">{item.available_quantity}kg</Typography>
        </View>
      </View>
      
      {bids[item.$id] && bids[item.$id].length > 0 ? (
        <View style={styles.bidsSection}>
          <Typography variant="body" style={styles.bidsSectionTitle}>
            Bids ({bids[item.$id].length})
          </Typography>
          {bids[item.$id].map(bid => renderBid(bid, item.$id))}
        </View>
      ) : (
        <Typography variant="body" color="textSecondary" style={styles.noBidsText}>
          No bids yet
        </Typography>
      )}
    </Card>
  )
  
  const renderAddForm = () => (
    <Card variant="elevated" style={styles.formCard}>
      <Typography variant="bodyLarge" style={styles.formTitle}>Create New Listing</Typography>
      
      <View style={styles.formField}>
        <Typography variant="body">Crop Type</Typography>
        <TextInput
          style={[styles.input, { 
            backgroundColor: colors.backgroundSecondary,
            borderColor: colors.border,
            color: colors.text
          }]}
          value={formData.crop_type}
          onChangeText={text => setFormData({...formData, crop_type: text})}
          placeholder="e.g., Wheat, Rice, etc."
          placeholderTextColor={colors.textSecondary}
        />
      </View>
      
      <View style={styles.formField}>
        <Typography variant="body">Price per kg (₹)</Typography>
        <TextInput
          style={[styles.input, { 
            backgroundColor: colors.backgroundSecondary,
            borderColor: colors.border,
            color: colors.text
          }]}
          value={formData.price_per_kg}
          onChangeText={text => setFormData({...formData, price_per_kg: text})}
          placeholder="e.g., 20.00"
          placeholderTextColor={colors.textSecondary}
          keyboardType="numeric"
        />
      </View>
      
      <View style={styles.formField}>
        <Typography variant="body">Total Quantity (kg)</Typography>
        <TextInput
          style={[styles.input, { 
            backgroundColor: colors.backgroundSecondary,
            borderColor: colors.border,
            color: colors.text
          }]}
          value={formData.total_quantity}
          onChangeText={text => setFormData({...formData, total_quantity: text})}
          placeholder="e.g., 100.00"
          placeholderTextColor={colors.textSecondary}
          keyboardType="numeric"
        />
      </View>
      
      <View style={styles.formButtons}>
        <TouchableOpacity 
          style={[styles.formButton, { backgroundColor: colors.backgroundSecondary }]}
          onPress={() => setShowAddForm(false)}
        >
          <Typography variant="body">Cancel</Typography>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[styles.formButton, { backgroundColor: colors.primary }]}
          onPress={handleCreateListing}
        >
          <Typography variant="body" color="primary">Create</Typography>
        </TouchableOpacity>
      </View>
    </Card>
  )

  return (
    <View style={styles.container}>
      <View style={styles.headerActions}>
        <Typography variant="bodyLarge">My Crop Listings</Typography>
        <TouchableOpacity 
          style={[styles.addButton, { backgroundColor: colors.primary }]}
          onPress={() => setShowAddForm(true)}
        >
          <Plus width={20} height={20} stroke="white" />
          <Typography variant="body" color="textSecondary" style={[styles.addButtonText, { color: 'white' }]}>
            New Listing
          </Typography>
        </TouchableOpacity>
      </View>
      
      {showAddForm && renderAddForm()}
      
      {loading ? (
        <Typography variant="body" style={styles.loadingText}>Loading...</Typography>
      ) : (
        <FlatList
          data={listings}
          renderItem={renderListing}
          keyExtractor={(item) => item.$id}
          contentContainerStyle={styles.listContainer}
          ListEmptyComponent={
            <Typography variant="body" style={styles.emptyText}>
              You don't have any listings yet
            </Typography>
          }
        />
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  headerActions: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  addButton: {
    flexDirection: 'row',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
    alignItems: 'center',
  },
  addButtonText: {
    marginLeft: 8,
  },
  listContainer: {
    paddingBottom: 80,
  },
  loadingText: {
    textAlign: 'center',
    marginTop: 20,
  },
  emptyText: {
    textAlign: 'center',
    marginTop: 30,
  },
  listingCard: {
    marginBottom: 16,
    padding: 16,
  },
  listingHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  listingTitleSection: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statusBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    marginLeft: 8,
  },
  cancelButton: {
    width: 36,
    height: 36,
    borderRadius: 18,
    borderWidth: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  listingDetails: {
    marginBottom: 12,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 4,
  },
  bidsSection: {
    marginTop: 12,
  },
  bidsSectionTitle: {
    fontWeight: '600',
    marginBottom: 8,
  },
  bidItem: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: 12,
    borderWidth: 1,
    borderRadius: 8,
    marginBottom: 8,
  },
  bidActions: {
    flexDirection: 'row',
  },
  bidActionButton: {
    paddingHorizontal: 10,
    paddingVertical: 6,
    borderRadius: 4,
    marginLeft: 8,
  },
  noBidsText: {
    marginTop: 12,
    textAlign: 'center',
  },
  formCard: {
    padding: 16,
    marginBottom: 16,
  },
  formTitle: {
    fontWeight: '600',
    marginBottom: 16,
  },
  formField: {
    marginBottom: 12,
  },
  input: {
    borderWidth: 1,
    borderRadius: 8,
    height: 40,
    paddingHorizontal: 12,
    marginTop: 4,
  },
  formButtons: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
    marginTop: 16,
  },
  formButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 8,
    marginLeft: 12,
  },
})

export default MarketListings
