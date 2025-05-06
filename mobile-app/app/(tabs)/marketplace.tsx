import { View, StyleSheet, FlatList, Image, TouchableOpacity, TextInput, ScrollView, Text } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../../context/ThemeContext"
import { useState } from "react"
import { Card } from "../../components/Card"
import { Typography } from "../../components/Typography"
import { Search, Filter, MessageSquare, DollarSign, Tag, Heart } from "react-native-feather"
import MarketListings from "../../components/marketplace/MarketListings"
import MarketContracts from "../../components/marketplace/MarketContracts"
import { TranslatedText } from "../../components/TranslatedText"

type MarketItem = {
  id: string
  title: string
  description: string
  price: number
  category: string
  seller: {
    name: string
    location: string
    rating: number
    image: string
  }
  image: string
  postedDate: string
  bids: { id: string, amount: number, user: string }[]
  isFavorite: boolean
}

export default function MarketplaceScreen() {
  const { colors, spacing, radius } = useTheme()
  const [activeTab, setActiveTab] = useState("listings")
  const [searchQuery, setSearchQuery] = useState("")
  const [marketItems, setMarketItems] = useState<MarketItem[]>([
    {
      id: "1",
      title: "Organic Fertilizer - 50kg",
      description: "High-quality organic fertilizer suitable for all crops. Enhances soil fertility and promotes healthy plant growth.",
      price: 45,
      category: "Supplies",
      seller: {
        name: "Eco Farming Co.",
        location: "Chianti Hills",
        rating: 4.8,
        image: "https://images.unsplash.com/photo-1560250097-0b93528c311a?ixlib=rb-4.0.3",
      },
      image: "https://images.unsplash.com/photo-1605000797499-95a51c5269ae?ixlib=rb-4.0.3",
      postedDate: "2024-04-28",
      bids: [
        { id: "b1", amount: 42, user: "GreenFarm" },
        { id: "b2", amount: 43, user: "AgriTech" }
      ],
      isFavorite: false
    },
    {
      id: "2",
      title: "Tractor for Rent - Daily/Weekly",
      description: "John Deere 5075E utility tractor available for rent. Perfect for small to medium farms. Rates negotiable for long-term rental.",
      price: 120,
      category: "Equipment",
      seller: {
        name: "Farm Machinery Rental",
        location: "Montalcino",
        rating: 4.6,
        image: "https://images.unsplash.com/photo-1472099645785-5658abf4ff4e?ixlib=rb-4.0.3",
      },
      image: "https://images.unsplash.com/photo-1530267981375-f095d3f5217b?ixlib=rb-4.0.3",
      postedDate: "2024-04-25",
      bids: [
        { id: "b3", amount: 110, user: "ValleyFarms" }
      ],
      isFavorite: true
    },
    {
      id: "3",
      title: "Fresh Olives - Direct from Producer",
      description: "Premium quality olives freshly harvested from our organic farm. Great for pressing or table use. Minimum order 25kg.",
      price: 8,
      category: "Produce",
      seller: {
        name: "Tuscany Olive Groves",
        location: "San Gimignano",
        rating: 4.9,
        image: "https://images.unsplash.com/photo-1580489944761-15a19d654956?ixlib=rb-4.0.3",
      },
      image: "https://images.unsplash.com/photo-1473649085228-583485e6e4d7?ixlib=rb-4.0.3",
      postedDate: "2024-04-30",
      bids: [],
      isFavorite: false
    }
  ])

  const [bidAmounts, setBidAmounts] = useState<{[key: string]: string}>({})

  const toggleFavorite = (id: string) => {
    setMarketItems(items => 
      items.map(item => 
        item.id === id ? {...item, isFavorite: !item.isFavorite} : item
      )
    )
  }

  const placeBid = (itemId: string) => {
    if (!bidAmounts[itemId] || isNaN(Number(bidAmounts[itemId]))) return
    
    const bidAmount = Number(bidAmounts[itemId])
    
    setMarketItems(items => 
      items.map(item => {
        if (item.id === itemId) {
          return {
            ...item,
            bids: [
              ...item.bids,
              { 
                id: `b${Date.now()}`, 
                amount: bidAmount, 
                user: "You" // In a real app, this would be the user's name
              }
            ]
          }
        }
        return item
      })
    )
    
    // Clear the bid input
    setBidAmounts({...bidAmounts, [itemId]: ""})
  }

  const renderTabContent = () => {
    switch (activeTab) {
      case "listings":
        return <MarketListings />
      case "contracts":
        return <MarketContracts />
      default:
        return <MarketListings />
    }
  }

  const renderMarketItem = ({ item }: { item: MarketItem }) => (
    <Card variant="elevated" style={styles.listingCard}>
      <Image source={{ uri: item.image }} style={styles.listingImage} />
      
      <View style={styles.listingContent}>
        <View style={styles.listingHeader}>
          <Typography variant="bodyLarge" style={{ fontWeight: '700' }}>{item.title}</Typography>
          <TouchableOpacity 
            style={styles.favoriteButton} 
            onPress={() => toggleFavorite(item.id)}
          >
            <Heart 
              width={20} 
              height={20} 
              stroke={item.isFavorite ? colors.error : colors.textSecondary} 
              fill={item.isFavorite ? colors.error : 'transparent'} 
            />
          </TouchableOpacity>
        </View>
        
        <View style={styles.categoryContainer}>
          <Tag width={14} height={14} stroke={colors.primary} />
          <Typography variant="small" color="primary" style={{ marginLeft: 4, fontWeight: '600' }}>
            {item.category}
          </Typography>
        </View>
        
        <Typography variant="caption" color="textSecondary" style={styles.description}>
          {item.description}
        </Typography>
        
        <View style={styles.sellerContainer}>
          <Image source={{ uri: item.seller.image }} style={styles.sellerImage} />
          <View style={{ flex: 1 }}>
            <Typography variant="caption" style={{ fontWeight: '600' }}>{item.seller.name}</Typography>
            <Typography variant="small" color="textSecondary">{item.seller.location}</Typography>
          </View>
          <Typography variant="bodyLarge" color="primary" style={{ fontWeight: '700' }}>
            ${item.price}
          </Typography>
        </View>
        
        <View style={styles.bidSection}>
          <Typography variant="caption" style={styles.bidSectionTitle}>
            {item.bids.length > 0 ? `Current Bids (${item.bids.length})` : "No bids yet"}
          </Typography>
          
          {item.bids.length > 0 && (
            <View style={styles.bidList}>
              {item.bids.slice(0, 2).map((bid, index) => (
                <View key={bid.id} style={styles.bidItem}>
                  <Typography variant="small" color="textSecondary">{bid.user}</Typography>
                  <Typography variant="small" color="primary">${bid.amount}</Typography>
                </View>
              ))}
              {item.bids.length > 2 && (
                <Typography variant="small" color="textSecondary">
                  {`+${(item.bids.length - 2).toString()} more bids`}
                </Typography>
              )}
            </View>
          )}
          
          <View style={styles.placeBidContainer}>
            <TextInput
              style={[
                styles.bidInput, 
                { 
                  backgroundColor: colors.backgroundSecondary,
                  borderColor: colors.border,
                  color: colors.text
                }
              ]}
              placeholder="Enter your bid..."
              placeholderTextColor={colors.textSecondary}
              keyboardType="numeric"
              value={bidAmounts[item.id] || ''}
              onChangeText={(text) => setBidAmounts({...bidAmounts, [item.id]: text})}
            />
            <TouchableOpacity 
              style={[styles.bidButton, { backgroundColor: colors.primary }]}
              onPress={() => placeBid(item.id)}
            >
              <DollarSign width={18} height={18} stroke="white" />
            </TouchableOpacity>
          </View>
        </View>
        
        <TouchableOpacity style={[styles.contactButton, { borderColor: colors.border }]}>
          <MessageSquare width={16} height={16} stroke={colors.primary} />
          <Typography variant="small" color="primary" style={{ marginLeft: 6 }}>
            Contact Seller
          </Typography>
        </TouchableOpacity>
      </View>
    </Card>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <View style={styles.header}>
        <Typography variant="heading">Marketplace</Typography>
        <TouchableOpacity 
          style={[
            styles.filterButton, 
            { 
              backgroundColor: colors.backgroundSecondary,
            }
          ]}
        >
          <Filter width={20} height={20} stroke={colors.primary} />
        </TouchableOpacity>
      </View>

      <View style={styles.tabsContainer}>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === "listings" && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab("listings")}
        >
          <Typography 
            variant="body" 
            color={activeTab === "listings" ? "primary" : "textSecondary"}
          >
            Listings
          </Typography>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === "contracts" && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab("contracts")}
        >
          <Typography 
            variant="body" 
            color={activeTab === "contracts" ? "primary" : "textSecondary"}
          >
            Contracts
          </Typography>
        </TouchableOpacity>
      </View>

      {renderTabContent()}
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 16,
  },
  filterButton: {
    width: 44,
    height: 44,
    borderRadius: 22,
    justifyContent: "center",
    alignItems: "center",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  searchContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginHorizontal: 20,
    marginBottom: 20,
    paddingHorizontal: 16,
    height: 54,
    borderRadius: 12,
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 1,
  },
  searchIcon: {
    marginRight: 10,
  },
  searchInput: {
    flex: 1,
    fontSize: 16,
  },
  listContainer: {
    padding: 20,
    paddingBottom: 100,
  },
  listingCard: {
    marginBottom: 24,
    overflow: "hidden",
    borderRadius: 16,
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  listingImage: {
    width: "100%",
    height: 200,
  },
  listingContent: {
    padding: 16,
  },
  listingHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 10,
  },
  favoriteButton: {
    width: 40,
    height: 40,
    borderRadius: 20,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },
  categoryContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
    backgroundColor: '#F0F8FF',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
    alignSelf: 'flex-start',
  },
  description: {
    marginBottom: 16,
    lineHeight: 20,
  },
  sellerContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
    backgroundColor: '#FAFAFA',
    padding: 10,
    borderRadius: 8,
  },
  sellerImage: {
    width: 40,
    height: 40,
    borderRadius: 20,
    marginRight: 12,
    borderWidth: 2,
    borderColor: 'white',
  },
  bidSection: {
    marginBottom: 16,
    backgroundColor: '#F8F9FA',
    padding: 16,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#EAEAEA',
  },
  bidSectionTitle: {
    fontWeight: "700",
    marginBottom: 12,
    fontSize: 15,
  },
  bidList: {
    marginBottom: 16,
    borderLeftWidth: 3,
    borderLeftColor: '#E0E7FF',
    paddingLeft: 12,
  },
  bidItem: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 8,
    paddingVertical: 4,
  },
  placeBidContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  bidInput: {
    flex: 1,
    height: 40,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 10,
    marginRight: 8,
  },
  bidButton: {
    width: 40,
    height: 40,
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
  },
  contactButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 10,
    borderWidth: 1,
    borderRadius: 8,
  },
  tabsContainer: {
    flexDirection: "row",
    paddingHorizontal: 20,
    borderBottomWidth: 1,
    marginBottom: 20,
  },
  tab: {
    paddingVertical: 12,
    marginRight: 24,
  },
  activeTab: {
    borderBottomWidth: 3,
  },
})
