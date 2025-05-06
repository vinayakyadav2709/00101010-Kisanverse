import React, { useState, useEffect } from 'react'
import { View, StyleSheet, FlatList, TouchableOpacity, Alert } from 'react-native'
import { useTheme } from '../../context/ThemeContext'
import { Typography } from '../Typography'
import { Card } from '../Card'
import { DollarSign, Check, X, AlertCircle, MapPin, Zap } from 'react-native-feather'
import { useUserStore } from '../../store/userStore'

type Subsidy = {
  $id: string
  program: string         // Name of the subsidy program
  description: string     // Description of the subsidy
  eligibility: string     // Criteria for eligibility
  type: string            // 'cash', 'asset', 'training', 'loan'
  benefits: string        // Value/benefit provided by the subsidy
  application_process: string // Application process details
  locations: string[]     // Locations where subsidy is applicable
  max_recipients: number  // Maximum number of recipients
  dynamic_fields: string  // Additional fields in JSON format
  provider: string        // Organization providing the subsidy
  status: string          // 'listed', 'removed', 'fulfilled'
  $createdAt: string
}

type SubsidyRequest = {
  $id: string
  subsidy_id: string
  farmer_id: string
  status: string
  $createdAt: string
  email?: string
}

const API_BASE_URL = 'https://4f70-124-66-175-40.ngrok-free.app';

const MarketSubsidies = () => {
  const { colors, spacing, radius } = useTheme()
  const { user } = useUserStore()
  const [subsidies, setSubsidies] = useState<Subsidy[]>([])
  const [requests, setRequests] = useState<SubsidyRequest[]>([])
  const [loading, setLoading] = useState(true)
  const [activeTab, setActiveTab] = useState('available')
  const [error, setError] = useState<string | null>(null)
  const [detailedError, setDetailedError] = useState<string | null>(null)

  // Fetch available subsidies and farmer's requests
  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        setError(null);
        setDetailedError(null);

        // Get farmer email from user store
        const farmerEmail = user?.email || "farmer@example.com";
        
        console.log('User from store:', user);
        console.log('Using farmer email from store:', farmerEmail);

        console.log('Fetching subsidies for email:', farmerEmail);
        // Use the email parameter to fetch subsidies 
        const subsidiesResponse = await fetch(`${API_BASE_URL}/subsidies?email=${farmerEmail}`);
        
        if (!subsidiesResponse.ok) {
          let errorMessage = `Failed to fetch subsidies: ${subsidiesResponse.status}`;
          
          try {
            // Try to extract detailed error message from response
            const errorData = await subsidiesResponse.json();
            if (errorData.message) {
              errorMessage = `Server error: ${errorData.message}`;
              setDetailedError(JSON.stringify(errorData, null, 2));
            }
          } catch (jsonError) {
            // If we couldn't parse JSON from error response
            console.log('Could not parse error response JSON:', jsonError);
          }
          
          throw new Error(errorMessage);
        }
        
        const subsidiesData = await subsidiesResponse.json();
        console.log('Subsidies data received:', subsidiesData);
        
        console.log('Fetching subsidy requests for email:', farmerEmail);
        const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${farmerEmail}`);
        
        if (!requestsResponse.ok) {
          let errorMessage = `Failed to fetch subsidy requests: ${requestsResponse.status}`;
          
          try {
            // Try to extract detailed error message from response
            const errorData = await requestsResponse.json();
            if (errorData.message) {
              errorMessage = `Server error: ${errorData.message}`;
              setDetailedError(JSON.stringify(errorData, null, 2));
            }
          } catch (jsonError) {
            // If we couldn't parse JSON from error response
            console.log('Could not parse error response JSON:', jsonError);
          }
          
          throw new Error(errorMessage);
        }
        
        const requestsData = await requestsResponse.json();
        console.log('Subsidy requests received:', requestsData);
        
        setSubsidies(subsidiesData.documents || []);
        setRequests(requestsData.documents || []);
        setLoading(false);
      } catch (error) {
        console.error('Error fetching data:', error);
        
        // Check if the error is related to network connectivity
        if (error instanceof TypeError && error.message.includes('Network request failed')) {
          setError('Network error: Unable to connect to server. Please check your internet connection.');
        } else if (error instanceof Error) {
          setError(error.message);
        } else {
          setError('An unknown error occurred while fetching data');
        }
        
        setLoading(false);
        
        // Fallback to mock data in case of API failure
        console.log('Using mock data as fallback');
        const mockSubsidies = [
          {
            $id: "1",
            program: "Seed Subsidy Program",
            description: "Provides high-quality seeds to farmers for wheat cultivation",
            eligibility: "Small and marginal farmers with land up to 5 acres",
            type: "asset",
            benefits: "Seeds worth ₹5,000 per acre",
            application_process: "Apply online with land records and Aadhaar",
            locations: ["Punjab", "Haryana"],
            max_recipients: 100,
            dynamic_fields: JSON.stringify({
              crop_type: "Wheat",
              seed_quantity: "25kg per acre",
              application_deadline: "30 days before sowing season"
            }),
            provider: "Ministry of Agriculture",
            status: "listed",
            $createdAt: "2025-05-01T12:00:00.000+00:00"
          },
          {
            $id: "2",
            program: "Farm Equipment Support",
            description: "Subsidy for purchase of modern farming equipment",
            eligibility: "All farmers with valid farming ID",
            type: "cash",
            benefits: "40% of cost, up to ₹2,00,000",
            application_process: "Apply at local agricultural office with equipment quotation",
            locations: ["All India"],
            max_recipients: 50,
            dynamic_fields: JSON.stringify({
              equipment_types: "Tractors, Harvesters, Irrigation systems",
              minimum_farm_size: "2 acres",
              farmer_contribution: "Minimum 20% of total cost"
            }),
            provider: "Agricultural Mechanization Scheme",
            status: "listed",
            $createdAt: "2025-05-05T10:30:00.000+00:00"
          }
        ];
        
        const mockRequests = [
          {
            $id: "sr1",
            subsidy_id: "3",
            farmer_id: "farmer_id",
            status: "requested",
            $createdAt: "2025-05-07T14:25:00.000+00:00"
          },
          {
            $id: "sr2",
            subsidy_id: "4",
            farmer_id: "farmer_id",
            status: "accepted",
            $createdAt: "2025-05-02T09:15:00.000+00:00"
          }
        ];
        
        setSubsidies(mockSubsidies);
        setRequests(mockRequests);
      }
    };
    
    fetchData();
  }, [user]);
  
  const handleRequestSubsidy = async (subsidyId: string) => {
    try {
      console.log('Requesting subsidy with ID:', subsidyId);
      
      // Get farmer information from store
      const farmerEmail = user?.email;
      
      if (!farmerEmail) {
        Alert.alert('Error', 'You must be logged in to request a subsidy');
        return;
      }
      
      console.log('Using farmer email from store:', farmerEmail);
      
      // Create the request body with only the required field according to the API
      const requestBody = { 
        subsidy_id: subsidyId,
        // Make sure we include the email in query parameters, not the body
      };
      
      console.log('Sending request with body:', requestBody);
      
      // Add email as a query parameter instead of in the body
      const response = await fetch(`${API_BASE_URL}/subsidy_requests?email=${encodeURIComponent(farmerEmail)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(requestBody)
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response from server:', errorData);
        throw new Error(errorData.message || 'Failed to submit subsidy request');
      }
      
      const data = await response.json();
      console.log('Subsidy request created successfully:', data);
      
      // Update the local state
      setRequests([...requests, data]);
      Alert.alert('Success', 'Subsidy request submitted successfully');
      
      // Refresh the requests list
      const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${farmerEmail}`);
      const requestsData = await requestsResponse.json();
      setRequests(requestsData.documents || []);
    } catch (error) {
      console.error('Error requesting subsidy:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to submit subsidy request');
    }
  };
  
  const handleWithdrawRequest = async (requestId: string) => {
    try {
      console.log('Withdrawing request with ID:', requestId);
      
      // Get farmer email from store
      const farmerEmail = user?.email;
      
      if (!farmerEmail) {
        Alert.alert('Error', 'You must be logged in to withdraw a request');
        return;
      }
      
      console.log('Using farmer email from store:', farmerEmail);
      
      // Move email to query parameter instead of request body
      const response = await fetch(`${API_BASE_URL}/subsidy_requests/${requestId}?email=${encodeURIComponent(farmerEmail)}`, {
        method: 'DELETE',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response from server:', errorData);
        throw new Error(errorData.message || 'Failed to withdraw request');
      }
      
      const data = await response.json();
      console.log('Request withdrawn successfully:', data);
      
      // Update local state
      setRequests(requests.map(req => 
        req.$id === requestId ? { ...req, status: 'withdrawn' } : req
      ));
      
      Alert.alert('Success', 'Request withdrawn successfully');
      
      // Refresh the requests list
      const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${farmerEmail}`);
      const requestsData = await requestsResponse.json();
      setRequests(requestsData.documents || []);
    } catch (error) {
      console.error('Error withdrawing request:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to withdraw request');
    }
  };
  
  const handleFulfillRequest = async (requestId: string) => {
    try {
      console.log('Marking request as fulfilled with ID:', requestId);
      
      // Get farmer email from store
      const farmerEmail = user?.email;
      
      if (!farmerEmail) {
        Alert.alert('Error', 'You must be logged in to mark a subsidy as fulfilled');
        return;
      }
      
      console.log('Using farmer email from store:', farmerEmail);
      
      // Move email to query parameter instead of request body
      const response = await fetch(`${API_BASE_URL}/subsidy_requests/${requestId}/fulfill?email=${encodeURIComponent(farmerEmail)}`, {
        method: 'PATCH',
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        console.error('Error response from server:', errorData);
        throw new Error(errorData.message || 'Failed to mark subsidy as fulfilled');
      }
      
      const data = await response.json();
      console.log('Subsidy marked as fulfilled successfully:', data);
      
      // Update local state
      setRequests(requests.map(req => 
        req.$id === requestId ? { ...req, status: 'fulfilled' } : req
      ));
      
      Alert.alert('Success', 'Subsidy marked as fulfilled');
      
      // Refresh the requests list
      const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${farmerEmail}`);
      const requestsData = await requestsResponse.json();
      setRequests(requestsData.documents || []);
    } catch (error) {
      console.error('Error fulfilling subsidy:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to mark subsidy as fulfilled');
    }
  };
  
  const renderDynamicFields = (fieldsJson: string) => {
    try {
      const fields = JSON.parse(fieldsJson || '{}');
      return Object.entries(fields).map(([key, value]) => {
        // Ensure key is a string and handle capitalization safely
        const formattedKey = (key || '')
          .split('_')
          .filter(Boolean) // Remove empty segments
          .map(word => {
            if (!word) return '';
            return word.charAt(0).toUpperCase() + word.slice(1).toLowerCase();
          })
          .join(' ');
        
        return (
          <View key={key || Math.random().toString()} style={styles.fieldRow}>
            <Typography variant="body" style={styles.fieldLabel}>
              {formattedKey || 'Field'}:
            </Typography>
            <Typography variant="body">{value !== undefined ? String(value) : ''}</Typography>
          </View>
        );
      });
    } catch (e) {
      console.error('Error parsing dynamic fields:', e);
      return <Typography variant="body" color="error">Invalid subsidy details</Typography>
    }
  }
  
  const getStatusColor = (status: string) => {
    switch (status) {
      case 'listed':
      case 'accepted':
        return colors.success
      case 'requested':
        return colors.warning
      case 'fulfilled':
        return colors.primary
      default:
        return colors.error
    }
  }
  
  const renderSubsidy = ({ item }: { item: Subsidy }) => (
    <Card variant="elevated" style={styles.subsidyCard}>
      <View style={styles.subsidyHeader}>
        <View style={styles.subsidyTitleContainer}>
          <View style={[styles.iconContainer, { backgroundColor: colors.primary + '10' }]}>
            <DollarSign width={20} height={20} stroke={colors.primary} />
          </View>
          <Typography variant="bodyLarge" style={styles.subsidyTitle}>
            {item.program}
          </Typography>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(item.status) + '20' }]}>
          <Typography variant="small" style={{ color: getStatusColor(item.status), fontWeight: '600' }}>
            {item.status.toUpperCase()}
          </Typography>
        </View>
      </View>
      
      <Typography variant="body" style={styles.provider}>
        <Typography variant="body" style={{ fontWeight: '600' }}>Provider:</Typography> {item.provider}
      </Typography>
      
      <Typography variant="body" style={styles.description}>
        {item.description}
      </Typography>
      
      <View style={styles.divider} />
      
      <View style={styles.benefitContainer}>
        <Typography variant="body" style={styles.benefitLabel}>
          Benefit:
        </Typography>
        <Typography variant="body" style={styles.benefitValue}>
          {item.benefits}
        </Typography>
      </View>
      
      <View style={styles.locations}>
        <MapPin width={16} height={16} stroke={colors.textSecondary} />
        <Typography variant="caption" color="textSecondary" style={{ marginLeft: 8 }}>
          Available in: <Typography variant="caption" style={{ fontWeight: '600' }}>
            {item.locations.includes("All India") ? "All India" : item.locations.join(', ')}
          </Typography>
        </Typography>
      </View>
      
      <View style={[styles.subsidyDetails, { borderLeftColor: colors.primary }]}>
        <Typography variant="body" style={styles.fieldLabel}>
          Eligibility:
        </Typography>
        <Typography variant="body" style={styles.fieldValue}>
          {item.eligibility}
        </Typography>
        
        <View style={styles.divider} />
        
        {renderDynamicFields(item.dynamic_fields)}
      </View>
      
      <View style={styles.recipientsInfo}>
        <Zap width={16} height={16} stroke={colors.warning} />
        <Typography variant="caption" color="textSecondary" style={{ marginLeft: 8 }}>
          Limited to <Typography variant="caption" style={{ fontWeight: '700', color: colors.warning }}>{item.max_recipients}</Typography> recipients
        </Typography>
      </View>
      
      <TouchableOpacity 
        style={[styles.actionButton, { backgroundColor: colors.primary }]}
        activeOpacity={0.8}
        onPress={() => handleRequestSubsidy(item.$id)}
      >
        <Typography variant="body" style={{ color: 'white', fontWeight: '600' }}>Apply for Subsidy</Typography>
      </TouchableOpacity>
    </Card>
  )
  
  const renderRequest = ({ item }: { item: SubsidyRequest }) => (
    <Card variant="elevated" style={styles.requestCard}>
      <View style={styles.requestHeader}>
        <View style={styles.requestIdContainer}>
          <View style={[styles.requestIconContainer, { backgroundColor: colors.primary + '10' }]}>
            <DollarSign width={16} height={16} stroke={colors.primary} />
          </View>
          <Typography variant="body" style={styles.requestId}>
            Application #{item.$id.slice(0, 6)}
          </Typography>
        </View>
        <View style={[styles.statusBadge, { backgroundColor: getStatusColor(item.status) + '20' }]}>
          <Typography variant="small" style={{ color: getStatusColor(item.status), fontWeight: '600' }}>
            {item.status.toUpperCase()}
          </Typography>
        </View>
      </View>
      
      <Typography variant="caption" color="textSecondary" style={styles.requestDate}>
        Applied on: <Typography variant="caption" style={{ fontWeight: '600' }}>{new Date(item.$createdAt).toLocaleDateString()}</Typography>
      </Typography>
      
      <View style={styles.requestActions}>
        {item.status === 'requested' && (
          <TouchableOpacity 
            style={[styles.requestActionButton, { backgroundColor: colors.error + '20' }]}
            activeOpacity={0.7}
            onPress={() => handleWithdrawRequest(item.$id)}
          >
            <X width={16} height={16} stroke={colors.error} />
            <Typography variant="small" style={{ color: colors.error, marginLeft: 6, fontWeight: '600' }}>
              Withdraw Application
            </Typography>
          </TouchableOpacity>
        )}
        
        {item.status === 'accepted' && (
          <TouchableOpacity 
            style={[styles.requestActionButton, { backgroundColor: colors.primary + '20' }]}
            activeOpacity={0.7}
            onPress={() => handleFulfillRequest(item.$id)}
          >
            <Check width={16} height={16} stroke={colors.primary} />
            <Typography variant="small" color="primary" style={{ marginLeft: 6, fontWeight: '600' }}>
              Mark as Received
            </Typography>
          </TouchableOpacity>
        )}
      </View>
    </Card>
  )

  return (
    <View style={styles.container}>
      <View style={styles.tabs}>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === 'available' && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab('available')}
        >
          <Typography 
            variant="body" 
            color={activeTab === 'available' ? 'primary' : 'textSecondary'}
            style={{ fontWeight: activeTab === 'available' ? '600' : '400' }}
          >
            Available Subsidies
          </Typography>
        </TouchableOpacity>
        <TouchableOpacity 
          style={[
            styles.tab, 
            activeTab === 'applications' && [styles.activeTab, { borderBottomColor: colors.primary }]
          ]}
          onPress={() => setActiveTab('applications')}
        >
          <Typography 
            variant="body" 
            color={activeTab === 'applications' ? 'primary' : 'textSecondary'}
            style={{ fontWeight: activeTab === 'applications' ? '600' : '400' }}
          >
            My Applications
          </Typography>
        </TouchableOpacity>
      </View>
      
      {loading ? (
        <View style={styles.loadingContainer}>
          <Typography variant="body" style={styles.loadingText}>Loading...</Typography>
        </View>
      ) : error ? (
        <View style={styles.errorContainer}>
          <AlertCircle width={32} height={32} stroke={colors.error} />
          <Typography variant="body" color="error" style={styles.errorText}>
            {error}
          </Typography>
          {detailedError && (
            <View style={styles.detailedErrorContainer}>
              <Typography variant="caption" color="textSecondary" style={styles.errorSubtext}>
                Technical details:
              </Typography>
              <Typography variant="caption" style={styles.detailedErrorText}>
                {detailedError}
              </Typography>
            </View>
          )}
          <Typography variant="caption" color="textSecondary" style={styles.errorSubtext}>
            Showing mock data instead
          </Typography>
        </View>
      ) : (
        activeTab === 'available' ? (
          <FlatList
            data={subsidies}
            renderItem={renderSubsidy}
            keyExtractor={(item) => item.$id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <AlertCircle width={32} height={32} stroke={colors.textSecondary} />
                <Typography variant="body" color="textSecondary" style={styles.emptyText}>
                  No subsidies available for your location
                </Typography>
              </View>
            }
          />
        ) : (
          <FlatList
            data={requests}
            renderItem={renderRequest}
            keyExtractor={(item) => item.$id}
            contentContainerStyle={styles.listContainer}
            showsVerticalScrollIndicator={false}
            ListEmptyComponent={
              <View style={styles.emptyContainer}>
                <AlertCircle width={32} height={32} stroke={colors.textSecondary} />
                <Typography variant="body" color="textSecondary" style={styles.emptyText}>
                  You haven't applied for any subsidies yet
                </Typography>
              </View>
            }
          />
        )
      )}
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  tabs: {
    flexDirection: 'row',
    marginBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  tab: {
    paddingVertical: 12,
    paddingHorizontal: 16,
    marginRight: 20,
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
    textAlign: 'center',
    marginTop: 20,
  },
  errorContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 40,
    padding: 20,
  },
  errorText: {
    marginTop: 16,
    textAlign: 'center',
    fontWeight: '600',
  },
  errorSubtext: {
    marginTop: 8,
    textAlign: 'center',
  },
  detailedErrorContainer: {
    marginTop: 12,
    padding: 12,
    borderRadius: 8,
    backgroundColor: '#f8f8f8',
    width: '100%',
    maxHeight: 150,
  },
  detailedErrorText: {
    fontFamily: 'monospace',
    fontSize: 12,
    color: '#666',
  },
  listContainer: {
    paddingBottom: 100,
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
    lineHeight: 22,
  },
  subsidyCard: {
    marginBottom: 20,
    padding: 16,
    borderRadius: 12,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 4,
  },
  subsidyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  subsidyTitleContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  iconContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    backgroundColor: '#E6F2FF', // Default light blue background
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
  },
  subsidyTitle: {
    fontWeight: '700',
    fontSize: 18,
  },
  provider: {
    marginBottom: 12,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
  },
  locations: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 14,
    backgroundColor: '#F8F8F8',
    padding: 10,
    borderRadius: 8,
  },
  divider: {
    height: 1,
    backgroundColor: '#E0E0E0',
    marginVertical: 14,
  },
  subsidyDetails: {
    marginBottom: 18,
    backgroundColor: '#FAFAFA',
    padding: 14,
    borderRadius: 10,
    borderLeftWidth: 3,
    borderLeftColor: '#2563EB', // Default primary blue color
  },
  fieldRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  fieldLabel: {
    fontWeight: '600',
    marginRight: 8,
  },
  recipientsInfo: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 18,
    backgroundColor: '#FFF9E6',
    padding: 12,
    borderRadius: 8,
  },
  actionButton: {
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.15,
    shadowRadius: 3,
    elevation: 3,
  },
  requestCard: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
    shadowOffset: { width: 0, height: 3 },
    shadowOpacity: 0.1,
    shadowRadius: 6,
    elevation: 3,
  },
  requestHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  requestIdContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  requestIconContainer: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: '#E6F2FF', // Default light blue background
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 8,
  },
  requestId: {
    fontWeight: '600',
    marginLeft: 8,
  },
  requestDate: {
    marginBottom: 16,
  },
  requestActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  requestActionButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 20,
  },
  description: {
    marginBottom: 12,
    lineHeight: 20,
  },
  benefitContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    backgroundColor: '#F0F8FF',
    padding: 12,
    borderRadius: 8,
  },
  benefitLabel: {
    fontWeight: '600',
    marginRight: 8,
  },
  benefitValue: {
    flex: 1,
    fontWeight: '500',
  },
  fieldValue: {
    marginBottom: 8,
    lineHeight: 20,
  },
})

export default MarketSubsidies
