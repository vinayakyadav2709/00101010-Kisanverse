import React, { useState, useEffect } from 'react';
import {
  View,
  StyleSheet,
  SafeAreaView,
  ScrollView,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
  Dimensions
} from 'react-native';
import { useTheme } from '../../context/ThemeContext';
import { Typography } from '../../components/Typography';
import { Card } from '../../components/Card';
import { useUserStore } from '../../store/userStore';
import {
  ShoppingBag,
  Package,
  TrendingUp,
  Clock,
  DollarSign,
  Award,
  AlertCircle,
  ArrowRight,
  Check,
  X,
  Truck,
  LogOut
} from 'react-native-feather';
import { Link } from 'expo-router';

const API_BASE_URL = 'https://4f70-124-66-175-40.ngrok-free.app';
const screenWidth = Dimensions.get('window').width;

// Types
type Bid = {
  $id: string;
  quantity: number;
  price_per_kg: number;
  listing_id: string;
  buyer_id: string;
  status: string;
  $createdAt: string;
  $updatedAt: string;
  crop_type?: string; // Added from listing data
};

type MarketStatistic = {
  name: string;
  value: string;
  change: string;
  isPositive: boolean;
  icon: React.ReactNode;
};

export default function BuyerDashboardScreen() {
  const { colors, spacing, radius } = useTheme();
  const { user, logoutUser } = useUserStore();
  
  // State
  const [recentBids, setRecentBids] = useState<Bid[]>([]);
  const [marketStats, setMarketStats] = useState<MarketStatistic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch data
  useEffect(() => {
    fetchDashboardData();
  }, [user]);

  const fetchDashboardData = async () => {
    try {
      setLoading(true);
      setError(null);

      if (!user?.email) {
        throw new Error('User email not found');
      }

      // Fetch user's bids
      const bidsResponse = await fetch(`${API_BASE_URL}/bids?email=${user.email}`);
      if (!bidsResponse.ok) {
        throw new Error(`Failed to fetch bids: ${bidsResponse.status}`);
      }
      const bidsData = await bidsResponse.json();
      
      // Process bids - add crop types
      let processedBids = bidsData.documents || [];
      
      // Sort by creation date (most recent first) and limit to 5
      processedBids = processedBids
        .sort((a: Bid, b: Bid) => new Date(b.$createdAt).getTime() - new Date(a.$createdAt).getTime())
        .slice(0, 5);
      
      // Fetch crop type information for each bid
      for (const bid of processedBids) {
        try {
          const listingResponse = await fetch(`${API_BASE_URL}/listing/${bid.listing_id}`);
          if (listingResponse.ok) {
            const listingData = await listingResponse.json();
            bid.crop_type = listingData.crop_type;
          }
        } catch (error) {
          console.error(`Error fetching listing for bid ${bid.$id}:`, error);
          bid.crop_type = "Unknown Crop";
        }
      }
      
      setRecentBids(processedBids);
      
      // Setup mock market statistics data
      // In a real app, this would come from an API
      setMarketStats([
        {
          name: "Wheat Price",
          value: "₹23.50/kg",
          change: "+2.5%",
          isPositive: true,
          icon: <Package width={20} height={20} stroke={colors.primary} />
        },
        {
          name: "Rice Price",
          value: "₹36.75/kg",
          change: "-1.2%",
          isPositive: false,
          icon: <Package width={20} height={20} stroke={colors.warning} />
        },
        {
          name: "New Listings",
          value: "43",
          change: "+12",
          isPositive: true,
          icon: <TrendingUp width={20} height={20} stroke={colors.success} />
        },
        {
          name: "Avg. Delivery",
          value: "2.3 days",
          change: "-0.5 days",
          isPositive: true,
          icon: <Truck width={20} height={20} stroke={colors.primary} />
        }
      ]);
      
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      setLoading(false);
      
      // Use mock data as fallback
      setRecentBids([
        {
          $id: "bid1",
          quantity: 50.0,
          price_per_kg: 22.5,
          listing_id: "listing1",
          buyer_id: user?.$id || "buyer_id",
          status: "accepted",
          crop_type: "Wheat",
          $createdAt: new Date(Date.now() - 1000 * 60 * 60 * 24).toISOString(),
          $updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 12).toISOString()
        },
        {
          $id: "bid2",
          quantity: 100.0,
          price_per_kg: 35.0,
          listing_id: "listing2",
          buyer_id: user?.$id || "buyer_id",
          status: "pending",
          crop_type: "Rice",
          $createdAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString(),
          $updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 48).toISOString()
        },
        {
          $id: "bid3",
          quantity: 75.0,
          price_per_kg: 18.0,
          listing_id: "listing3",
          buyer_id: user?.$id || "buyer_id",
          status: "fulfilled",
          crop_type: "Corn",
          $createdAt: new Date(Date.now() - 1000 * 60 * 60 * 96).toISOString(),
          $updatedAt: new Date(Date.now() - 1000 * 60 * 60 * 72).toISOString()
        }
      ]);
      
      setMarketStats([
        {
          name: "Wheat Price",
          value: "₹23.50/kg",
          change: "+2.5%",
          isPositive: true,
          icon: <Package width={20} height={20} stroke={colors.primary} />
        },
        {
          name: "Rice Price",
          value: "₹36.75/kg",
          change: "-1.2%",
          isPositive: false,
          icon: <Package width={20} height={20} stroke={colors.warning} />
        },
        {
          name: "New Listings",
          value: "43",
          change: "+12",
          isPositive: true,
          icon: <TrendingUp width={20} height={20} stroke={colors.success} />
        },
        {
          name: "Avg. Delivery",
          value: "2.3 days",
          change: "-0.5 days",
          isPositive: true,
          icon: <Truck width={20} height={20} stroke={colors.primary} />
        }
      ]);
    }
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

  // Get status icon based on bid status
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'accepted':
        return <Check width={16} height={16} stroke={colors.success} />;
      case 'pending':
        return <Clock width={16} height={16} stroke={colors.warning} />;
      case 'rejected':
        return <X width={16} height={16} stroke={colors.error} />;
      case 'fulfilled':
        return <Award width={16} height={16} stroke={colors.primary} />;
      default:
        return <AlertCircle width={16} height={16} stroke={colors.textSecondary} />;
    }
  };

  // Get color based on bid status
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'accepted': return colors.success;
      case 'pending': return colors.warning;
      case 'rejected': return colors.error;
      case 'fulfilled': return colors.primary;
      default: return colors.textSecondary;
    }
  };

  // Render a recent bid item
  const renderBidItem = ({ item }: { item: Bid }) => (
    <Card variant="elevated" style={styles.bidCard}>
      <View style={styles.bidHeader}>
        <View style={styles.cropInfo}>
          <View style={[styles.cropIconContainer, { backgroundColor: colors.primary + '15' }]}>
            <Package width={18} height={18} stroke={colors.primary} />
          </View>
          <View>
            <Typography variant="bodyLarge" style={styles.cropType}>
              {item.crop_type || 'Unknown Crop'}
            </Typography>
            <Typography variant="caption" color="textSecondary">
              {formatDate(item.$createdAt)}
            </Typography>
          </View>
        </View>
        <View style={[
          styles.statusBadge, 
          { backgroundColor: getStatusColor(item.status) + '20' }
        ]}>
          <View style={styles.statusContent}>
            {getStatusIcon(item.status)}
            <Typography 
              variant="small" 
              style={{ 
                color: getStatusColor(item.status), 
                fontWeight: '600', 
                marginLeft: 4 
              }}
            >
              {item.status.toUpperCase()}
            </Typography>
          </View>
        </View>
      </View>
      
      <View style={styles.bidDetails}>
        <View style={styles.bidDetailRow}>
          <Typography variant="caption" color="textSecondary">
            Quantity:
          </Typography>
          <Typography variant="body" style={{ fontWeight: '500' }}>
            {item.quantity} kg
          </Typography>
        </View>
        
        <View style={styles.bidDetailRow}>
          <Typography variant="caption" color="textSecondary">
            Price:
          </Typography>
          <Typography variant="body" style={{ fontWeight: '500' }}>
            ₹{item.price_per_kg}/kg
          </Typography>
        </View>
        
        <View style={styles.bidDetailRow}>
          <Typography variant="caption" color="textSecondary">
            Total Value:
          </Typography>
          <Typography variant="bodyLarge" style={{ fontWeight: '700', color: colors.text }}>
            ₹{(item.quantity * item.price_per_kg).toLocaleString()}
          </Typography>
        </View>
      </View>
    </Card>
  );

  // Render a market statistic card
  const renderStatItem = ({ item }: { item: MarketStatistic }) => (
    <Card variant="elevated" style={styles.statCard}>
      <View style={styles.statHeader}>
        <View style={[styles.statIconContainer, { backgroundColor: colors.background }]}>
          {item.icon}
        </View>
        <Typography variant="caption" color="textSecondary" style={styles.statName}>
          {item.name}
        </Typography>
      </View>
      
      <Typography variant="subheading" style={styles.statValue}>
        {item.value}
      </Typography>
      
      <View style={[
        styles.statChangeBadge, 
        { backgroundColor: item.isPositive ? colors.success + '15' : colors.error + '15' }
      ]}>
        <Typography 
          variant="small" 
          style={{ 
            color: item.isPositive ? colors.success : colors.error,
            fontWeight: '600'
          }}
        >
          {item.change}
        </Typography>
      </View>
    </Card>
  );

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView
        showsVerticalScrollIndicator={false}
        contentContainerStyle={styles.scrollContent}
      >
        <View style={styles.header}>
          <View>
            <Typography variant="heading" style={styles.greeting}>
              Hello, {user?.name?.split(' ')[0] || 'Buyer'}
            </Typography>
            <Typography variant="body" color="textSecondary">
              Welcome to your marketplace dashboard
            </Typography>
          </View>
          <View style={{flexDirection: 'row'}}>
            <TouchableOpacity
              onPress={logoutUser}
              style={[styles.logoutButton, { backgroundColor: colors.error + '15', marginRight: 8 }]}
            >
              <LogOut width={20} height={20} stroke={colors.error} />
            </TouchableOpacity>
            <TouchableOpacity
              style={[styles.refreshButton, { backgroundColor: colors.primary + '15' }]}
              onPress={fetchDashboardData}
            >
              <Typography variant="small" color="primary">Refresh</Typography>
            </TouchableOpacity>
          </View>
        </View>

        {/* Market Statistics */}
        <View style={styles.section}>
          <Typography variant="subheading" style={styles.sectionTitle}>
            Market Insights
          </Typography>
          <Typography variant="body" color="textSecondary" style={styles.sectionSubtitle}>
            Current market trends and statistics
          </Typography>
          
          {loading ? (
            <ActivityIndicator size="large" color={colors.primary} style={styles.loader} />
          ) : error ? (
            <View style={styles.errorContainer}>
              <AlertCircle width={24} height={24} stroke={colors.error} />
              <Typography variant="body" color="error" style={styles.errorText}>
                {error}
              </Typography>
            </View>
          ) : (
            <FlatList
              data={marketStats}
              renderItem={renderStatItem}
              keyExtractor={(item) => item.name}
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.statsContainer}
              ItemSeparatorComponent={() => <View style={{ width: 12 }} />}
            />
          )}
        </View>
        
        {/* Recent Orders */}
        <View style={styles.section}>
          <View style={styles.sectionHeaderRow}>
            <View>
              <Typography variant="subheading" style={styles.sectionTitle}>
                Recent Orders
              </Typography>
              <Typography variant="body" color="textSecondary" style={styles.sectionSubtitle}>
                Your latest bids and purchases
              </Typography>
            </View>
            <Link href="/(buyer-tabs)/marketplace" asChild>
              <TouchableOpacity style={styles.viewAllButton}>
                <Typography variant="body" color="primary" style={{ fontWeight: '600' }}>
                  View All
                </Typography>
                <ArrowRight width={16} height={16} stroke={colors.primary} style={{ marginLeft: 4 }} />
              </TouchableOpacity>
            </Link>
          </View>
          
          {loading ? (
            <ActivityIndicator size="large" color={colors.primary} style={styles.loader} />
          ) : error ? (
            <View style={styles.errorContainer}>
              <AlertCircle width={24} height={24} stroke={colors.error} />
              <Typography variant="body" color="error" style={styles.errorText}>
                {error}
              </Typography>
            </View>
          ) : recentBids.length === 0 ? (
            <Card variant="elevated" style={styles.emptyStateCard}>
              <ShoppingBag width={32} height={32} stroke={colors.textSecondary} />
              <Typography variant="body" color="textSecondary" style={styles.emptyStateText}>
                You haven't placed any bids yet
              </Typography>
              <Link href="/(buyer-tabs)/marketplace" asChild>
                <TouchableOpacity 
                  style={[styles.browseCropsButton, { backgroundColor: colors.primary }]}
                >
                  <Typography variant="body" style={{ color: 'white', fontWeight: '600' }}>
                    Browse Available Crops
                  </Typography>
                </TouchableOpacity>
              </Link>
            </Card>
          ) : (
            <FlatList
              data={recentBids}
              renderItem={renderBidItem}
              keyExtractor={(item) => item.$id}
              scrollEnabled={false}
              contentContainerStyle={styles.bidsContainer}
              ItemSeparatorComponent={() => <View style={{ height: 12 }} />}
            />
          )}
        </View>
        
        {/* Quick Stats */}
        <View style={styles.section}>
          <Typography variant="subheading" style={styles.sectionTitle}>
            Your Purchase Stats
          </Typography>
          
          <View style={styles.statsGrid}>
            <Card variant="elevated" style={[styles.statGridCard, { backgroundColor: colors.primary + '10' }]}>
              <DollarSign width={24} height={24} stroke={colors.primary} />
              <Typography variant="subheading" style={[styles.statGridValue, { color: colors.primary }]}>
                ₹{(recentBids.reduce((sum, bid) => sum + (bid.quantity * bid.price_per_kg), 0)).toLocaleString()}
              </Typography>
              <Typography variant="small" color="textSecondary">
                Total Purchases
              </Typography>
            </Card>
            
            <Card variant="elevated" style={[styles.statGridCard, { backgroundColor: colors.success + '10' }]}>
              <Package width={24} height={24} stroke={colors.success} />
              <Typography variant="subheading" style={[styles.statGridValue, { color: colors.success }]}>
                {recentBids.reduce((sum, bid) => sum + bid.quantity, 0).toLocaleString()} kg
              </Typography>
              <Typography variant="small" color="textSecondary">
                Total Quantity
              </Typography>
            </Card>
            
            <Card variant="elevated" style={[styles.statGridCard, { backgroundColor: colors.warning + '10' }]}>
              <Clock width={24} height={24} stroke={colors.warning} />
              <Typography variant="subheading" style={[styles.statGridValue, { color: colors.warning }]}>
                {recentBids.filter(bid => bid.status === 'pending').length}
              </Typography>
              <Typography variant="small" color="textSecondary">
                Pending Bids
              </Typography>
            </Card>
            
            <Card variant="elevated" style={[styles.statGridCard, { backgroundColor: colors.info + '10' }]}>
              <Award width={24} height={24} stroke={colors.info} />
              <Typography variant="subheading" style={[styles.statGridValue, { color: colors.info }]}>
                {recentBids.filter(bid => bid.status === 'fulfilled').length}
              </Typography>
              <Typography variant="small" color="textSecondary">
                Completed Orders
              </Typography>
            </Card>
          </View>
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  scrollContent: {
    padding: 20,
    paddingBottom: 80,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  logoutButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  greeting: {
    fontSize: 28,
    fontWeight: '700',
  },
  refreshButton: {
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 20,
  },
  section: {
    marginBottom: 24,
  },
  sectionHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 4,
  },
  sectionSubtitle: {
    marginBottom: 16,
  },
  viewAllButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  statsContainer: {
    paddingRight: 20,
  },
  statCard: {
    width: screenWidth * 0.4,
    padding: 16,
    marginBottom: 4,
  },
  statHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
  },
  statIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  statName: {
    flex: 1,
  },
  statValue: {
    fontSize: 20,
    fontWeight: '700',
    marginBottom: 8,
  },
  statChangeBadge: {
    alignSelf: 'flex-start',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 12,
  },
  bidsContainer: {
    paddingBottom: 8,
  },
  bidCard: {
    padding: 16,
    marginBottom: 4,
  },
  bidHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  cropInfo: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  cropIconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  cropType: {
    fontWeight: '600',
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 16,
  },
  statusContent: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  bidDetails: {
    marginTop: 4,
    backgroundColor: 'rgba(0, 0, 0, 0.02)',
    borderRadius: 8,
    padding: 12,
  },
  bidDetailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  loader: {
    marginTop: 24,
    marginBottom: 24,
  },
  errorContainer: {
    alignItems: 'center',
    padding: 24,
  },
  errorText: {
    marginTop: 12,
    textAlign: 'center',
  },
  emptyStateCard: {
    alignItems: 'center',
    padding: 24,
  },
  emptyStateText: {
    marginTop: 12,
    marginBottom: 20,
    textAlign: 'center',
  },
  browseCropsButton: {
    paddingHorizontal: 20,
    paddingVertical: 12,
    borderRadius: 8,
  },
  statsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  statGridCard: {
    width: '48%',
    padding: 16,
    marginBottom: 16,
    alignItems: 'center',
  },
  statGridValue: {
    fontSize: 20,
    fontWeight: '700',
    marginVertical: 8,
  },
});
