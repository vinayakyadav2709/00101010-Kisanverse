import { View, StyleSheet, TouchableOpacity, ActivityIndicator, Alert } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../../context/ThemeContext"
import { Typography } from "../../components/Typography"
import { useState, useEffect } from "react"
import { FlatList } from "react-native-gesture-handler"
import { Card } from "../../components/Card"
import { AlertCircle, Check, Zap } from "react-native-feather"
import { useUserStore } from "../../store/userStore"

const API_BASE_URL = "https://4f70-124-66-175-40.ngrok-free.app";

type Subsidy = {
  $id: string;
  program: string;
  description: string;
  eligibility: string;
  type: string;
  benefits: string;
  application_process: string;
  locations: string[];
  max_recipients: number;
  dynamic_fields: string;
  provider: string;
  status: string;
  $createdAt: string;
};

type SubsidyRequest = {
  $id: string;
  subsidy_id: string;
  farmer_id: string;
  status: string;
  request_date: string;
  program?: string;
  type?: string;
  benefits?: string;
};

export default function SchemeScreen() {
  const { colors } = useTheme();
  const { user } = useUserStore();
  
  // State
  const [activeTab, setActiveTab] = useState('available');
  const [subsidies, setSubsidies] = useState<Subsidy[]>([]);
  const [requests, setRequests] = useState<SubsidyRequest[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [detailedError, setDetailedError] = useState<string | null>(null);

  // Fetch subsidies and requests
  useEffect(() => {
    fetchSubsidiesAndRequests();
  }, [user]);

  const fetchSubsidiesAndRequests = async () => {
    try {
      setLoading(true);
      setError(null);
      setDetailedError(null);

      if (!user?.email) {
        throw new Error('User email not found');
      }

      console.log("[DEBUG] Fetching data for user:", user.email);
      console.log("[DEBUG] API Base URL:", API_BASE_URL);

      // Fetch available subsidies
      console.log("[DEBUG] Fetching subsidies...");
      const subsidiesResponse = await fetch(`${API_BASE_URL}/subsidies`);
      console.log("[DEBUG] Subsidies response status:", subsidiesResponse.status);
      
      if (!subsidiesResponse.ok) {
        throw new Error(`Failed to fetch subsidies: ${subsidiesResponse.status}`);
      }
      const subsidiesData = await subsidiesResponse.json();
      console.log("[DEBUG] Subsidies data:", JSON.stringify(subsidiesData).substring(0, 200) + "...");
      
      // Comment out the API fetch for subsidy requests
      /*
      // Fetch user's subsidy requests - update URL from subsidy-requests to subsidy_requests per API docs
      console.log("[DEBUG] Fetching subsidy requests...");
      const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${user.email}`);
      console.log("[DEBUG] Requests response status:", requestsResponse.status);
      
      if (!requestsResponse.ok) {
        throw new Error(`Failed to fetch subsidy requests: ${requestsResponse.status}`);
      }
      const requestsData = await requestsResponse.json();
      console.log("[DEBUG] Requests data:", JSON.stringify(requestsData).substring(0, 200) + "...");
      
      setSubsidies(subsidiesData.documents || []);
      setRequests(requestsData.documents || []);
      */
      
      // Set subsidies from API response
      setSubsidies(subsidiesData.documents || []);
      
      // Use hardcoded subsidy requests
      setRequests([
        {
          $id: "681764d9003d069dd006",
          subsidy_id: "681764d9003d069dd006",
          farmer_id: user?.$id || "user_id",
          status: "pending",
          request_date: "2025-05-01T12:00:00.000+00:00",
          program: "Crop Insurance Subsidy",
          type: "cash",
          benefits: "INR 5000"
        },
        {
          $id: "681764d90014aa473fc2",
          subsidy_id: "681764d90014aa473fc2",
          farmer_id: user?.$id || "user_id",
          status: "approved",
          request_date: "2025-04-20T12:00:00.000+00:00",
          program: "Drip Irrigation Scheme",
          type: "asset",
          benefits: "Free drip irrigation kit (upto 2 acres)"
        },
        {
          $id: "681764d90007dd985678",
          subsidy_id: "681764d90007dd985678",
          farmer_id: user?.$id || "user_id",
          status: "fulfilled",
          request_date: "2025-03-01T12:00:00.000+00:00",
          program: "Fertilizer Support Program",
          type: "cash",
          benefits: "INR 10,000"
        }
      ]);
      
      console.log("[DEBUG] Data fetching completed successfully");
      setLoading(false);
    } catch (error) {
      console.error('[DEBUG] Error fetching data:', error);
      console.error('[DEBUG] Error details:', error instanceof Error ? error.stack : 'No stack trace available');
      setError(error instanceof Error ? error.message : 'An unknown error occurred');
      setDetailedError(error instanceof Error ? error.stack ?? null : null);
      
      // Use mock data as fallback
      console.log("[DEBUG] Using mock data as fallback");
      setSubsidies([
        {
          $id: "1",
          program: "PM-KISAN Scheme",
          description: "Financial assistance to small and marginal farmers",
          eligibility: "Farmers with less than 2 hectares of land",
          type: "cash",
          benefits: "₹6,000 per year in three installments",
          application_process: "Apply through local agriculture office with land records",
          locations: ["All India"],
          max_recipients: 100,
          dynamic_fields: JSON.stringify({
            installment_dates: "Apr, Aug, Dec",
            documents_required: "Aadhaar, Land Records, Bank Details",
            helpline: "1800-XXX-XXX"
          }),
          provider: "Ministry of Agriculture & Farmers Welfare",
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
      ]);
      
      setRequests([
        {
          $id: "681764d9003d069dd006",
          subsidy_id: "681764d9003d069dd006",
          farmer_id: user?.$id || "user_id",
          status: "pending",
          request_date: "2025-05-01T12:00:00.000+00:00",
          program: "Crop Insurance Subsidy",
          type: "cash",
          benefits: "INR 5000"
        },
        {
          $id: "681764d90014aa473fc2",
          subsidy_id: "681764d90014aa473fc2",
          farmer_id: user?.$id || "user_id",
          status: "approved",
          request_date: "2025-04-20T12:00:00.000+00:00",
          program: "Drip Irrigation Scheme",
          type: "asset",
          benefits: "Free drip irrigation kit (upto 2 acres)"
        },
        {
          $id: "681764d90007dd985678",
          subsidy_id: "681764d90007dd985678",
          farmer_id: user?.$id || "user_id",
          status: "fulfilled",
          request_date: "2025-03-01T12:00:00.000+00:00",
          program: "Fertilizer Support Program",
          type: "cash",
          benefits: "INR 10,000"
        }
      ]);
      
      setLoading(false);
    }
  };

  // Request a subsidy
  const handleRequestSubsidy = async (subsidyId: string) => {
    try {
      const selectedSubsidy = subsidies.find(s => s.$id === subsidyId);
      if (!selectedSubsidy) {
        throw new Error('Subsidy not found');
      }
      
      // Check if user already requested this subsidy
      const alreadyRequested = requests.some(r => r.subsidy_id === subsidyId);
      if (alreadyRequested) {
        Alert.alert('Already Requested', 'You have already applied for this subsidy');
        return;
      }
      
      const response = await fetch(`${API_BASE_URL}/subsidy_requests?email=${user?.email}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          subsidy_id: subsidyId,
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to request subsidy');
      }
      
      // Add the new request to state
      const newRequest = {
        $id: `req${Date.now()}`,
        subsidy_id: subsidyId,
        farmer_id: user?.$id || "user_id",
        status: "pending",
        request_date: new Date().toISOString(),
        program: selectedSubsidy.program,
        type: selectedSubsidy.type,
        benefits: selectedSubsidy.benefits
      };
      
      setRequests([...requests, newRequest]);
      Alert.alert('Success', 'Your subsidy application has been submitted');
    } catch (error) {
      console.error('Error requesting subsidy:', error);
      Alert.alert('Error', error instanceof Error ? error.message : 'Failed to request subsidy');
    }
  };

  // Fulfill a subsidy request
  const handleFulfillRequest = async (requestId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/subsidy_requests/${requestId}?email=${user?.email}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          status: 'fulfilled'
        })
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.message || 'Failed to update request status');
      }
      
      // Update state with fulfilled status
      setRequests(requests.map(req => 
        req.$id === requestId ? {...req, status: 'fulfilled'} : req
      ));
      
      Alert.alert('Success', 'Subsidy marked as received');
      
      // Refresh data
      const requestsResponse = await fetch(`${API_BASE_URL}/subsidy_requests?email=${user?.email}`);
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
              {formattedKey}:
            </Typography>
            <Typography variant="body">
              {value as string}
            </Typography>
          </View>
        );
      });
    } catch (e) {
      return null;
    }
  };

  // Render a subsidy item
  const renderSubsidy = ({ item }: { item: Subsidy }) => (
    <Card variant="elevated" style={styles.subsidyCard}>
      <View style={styles.subsidyHeader}>
        <Typography variant="bodyLarge" style={styles.subsidyTitle}>
          {item.program}
        </Typography>
        <View style={[styles.typeBadge, { 
          backgroundColor: item.type === 'cash' ? '#10B981' + '20' : '#1D4ED8' + '20',
        }]}>
          <Typography 
            variant="small" 
            style={{ 
              color: item.type === 'cash' ? '#10B981' : '#1D4ED8',
              fontWeight: '600' 
            }}
          >
            {item.type.toUpperCase()}
          </Typography>
        </View>
      </View>
      
      <Typography variant="body" color="textSecondary" style={styles.subsidyDescription}>
        {item.description}
      </Typography>
      
      <View style={styles.subsidyDetails}>
        <Typography variant="body" style={styles.fieldLabel}>
          Benefits:
        </Typography>
        <Typography variant="body" style={styles.fieldValue}>
          {item.benefits}
        </Typography>
        
        <View style={styles.divider} />
        
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
  );

  // Render a request item
  const renderRequest = ({ item }: { item: SubsidyRequest }) => (
    <Card variant="elevated" style={styles.requestCard}>
      <View style={styles.requestHeader}>
        <Typography variant="bodyLarge" style={styles.requestTitle}>
          {item.program || 'Subsidy Program'}
        </Typography>
        <View style={[styles.statusBadge, { 
          backgroundColor: 
            item.status === 'approved' ? colors.success + '20' :
            item.status === 'rejected' ? colors.error + '20' :
            item.status === 'fulfilled' ? colors.primary + '20' :
            colors.warning + '20'
        }]}>
          <Typography 
            variant="small" 
            style={{ 
              color: 
                item.status === 'approved' ? colors.success :
                item.status === 'rejected' ? colors.error :
                item.status === 'fulfilled' ? colors.primary :
                colors.warning,
              fontWeight: '600' 
            }}
          >
            {item.status.toUpperCase()}
          </Typography>
        </View>
      </View>
      
      <View style={styles.requestDetails}>
        <View style={styles.fieldRow}>
          <Typography variant="body" style={styles.fieldLabel}>
            Benefits:
          </Typography>
          <Typography variant="body">
            {item.benefits || 'Unknown benefits'}
          </Typography>
        </View>
        
        <View style={styles.fieldRow}>
          <Typography variant="body" style={styles.fieldLabel}>
            Type:
          </Typography>
          <Typography variant="body">
            {item.type || 'Unknown type'}
          </Typography>
        </View>
        
        <View style={styles.fieldRow}>
          <Typography variant="body" style={styles.fieldLabel}>
            Applied On:
          </Typography>
          <Typography variant="body">
            {new Date(item.request_date).toLocaleDateString('en-US', {
              year: 'numeric',
              month: 'short',
              day: 'numeric'
            })}
          </Typography>
        </View>
      </View>
      
      {item.status === 'approved' && (
        <View style={styles.requestActions}>
          <TouchableOpacity 
            style={[styles.fulfillButton, { backgroundColor: colors.primary + '15' }]}
            activeOpacity={0.7}
            onPress={() => handleFulfillRequest(item.$id)}
          >
            <Check width={16} height={16} stroke={colors.primary} />
            <Typography variant="small" color="primary" style={{ marginLeft: 6, fontWeight: '600' }}>
              Mark as Received
            </Typography>
          </TouchableOpacity>
        </View>
      )}
    </Card>
  );

  return (
    <View style={styles.container}>
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <View style={styles.header}>
          <Typography variant="heading">Government Schemes</Typography>
        </View>
        
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
              Available Schemes
            </Typography>
          </TouchableOpacity>
          <TouchableOpacity 
            style={[
              styles.tab, 
              activeTab === 'requests' && [styles.activeTab, { borderBottomColor: colors.primary }]
            ]}
            onPress={() => setActiveTab('requests')}
          >
            <Typography 
              variant="body" 
              color={activeTab === 'requests' ? 'primary' : 'textSecondary'}
              style={{ fontWeight: activeTab === 'requests' ? '600' : '400' }}
            >
              My Applications
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
                    No subsidies available at the moment
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
      </SafeAreaView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    paddingHorizontal: 20,
    paddingTop: 20,
    paddingBottom: 16,
  },
  tabs: {
    flexDirection: 'row',
    marginBottom: 20,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
    paddingHorizontal: 20,
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
    marginTop: 16,
    width: '100%',
    padding: 12,
    backgroundColor: 'rgba(0,0,0,0.03)',
    borderRadius: 8,
  },
  detailedErrorText: {
    color: '#666',
    fontSize: 12,
    fontFamily: 'monospace',
  },
  emptyContainer: {
    alignItems: 'center',
    justifyContent: 'center',
    marginTop: 60,
  },
  emptyText: {
    marginTop: 16,
    textAlign: 'center',
  },
  listContainer: {
    padding: 20,
    paddingBottom: 80,
  },
  subsidyCard: {
    marginBottom: 24,
    padding: 16,
    borderRadius: 12,
  },
  subsidyHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  subsidyTitle: {
    flex: 1,
    fontSize: 18,
    fontWeight: '700',
    marginRight: 8,
  },
  typeBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  subsidyDescription: {
    marginBottom: 16,
    lineHeight: 20,
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
  fieldValue: {
    flex: 1,
  },
  recipientsInfo: {
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
  actionButton: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
  requestCard: {
    marginBottom: 16,
    padding: 16,
    borderRadius: 12,
  },
  requestHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  requestTitle: {
    flex: 1,
    fontWeight: '700',
    marginRight: 8,
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  requestDetails: {
    padding: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    borderRadius: 8,
    marginBottom: 16,
  },
  requestActions: {
    flexDirection: 'row',
    justifyContent: 'flex-end',
  },
  fulfillButton: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 14,
    paddingVertical: 8,
    borderRadius: 8,
  },
});
