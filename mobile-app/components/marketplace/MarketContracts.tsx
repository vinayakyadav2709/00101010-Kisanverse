import { View, StyleSheet, FlatList, TouchableOpacity, ScrollView } from "react-native";
import { useState } from "react";
import { Card } from "../Card";
import { Typography } from "../Typography";
import { Calendar, DollarSign, Package, Truck, Clock, Check, AlertCircle } from "react-native-feather";
import { useTheme } from "../../context/ThemeContext";

type ContractRequest = {
  $id: string;
  contract_id: string;
  farmer_id: string;
  status: string;
  $createdAt: string;
  $updatedAt: string;
};

type Contract = {
  $id: string;
  buyer_id: string;
  status: string;
  dynamic_fields: string;
  locations: string[];
  crop_type: string;
  quantity: number;
  price_per_kg: number;
  advance_payment: number;
  delivery_date: string;
  payment_terms: string;
  $createdAt: string;
  $updatedAt: string;
};

type ContractWithRequests = {
  details: Contract;
  requests: ContractRequest[];
};

const MarketContracts = () => {
  const { colors } = useTheme();
  const [activeFilter, setActiveFilter] = useState<string>("all");

  // Hardcoded contracts data
  const contractsData: Record<string, ContractWithRequests> = {
    "6817892000104e73ab7d": {
      details: {
        buyer_id: "681789200022c543786d",
        status: "accepted",
        dynamic_fields: "{\"requirements\": {\"grade_required\": \"A\", \"transport_required\": true}}",
        locations: [
          "CHHATTISGARH",
          "Delhi"
        ],
        crop_type: "RICE",
        quantity: 1000,
        price_per_kg: 25,
        advance_payment: 5000,
        delivery_date: "2025-10-31T15:34:56.473+00:00",
        payment_terms: "50% advance, 50% on delivery",
        $id: "6817892000104e73ab7d",
        $createdAt: "2025-05-04T15:34:56.478+00:00",
        $updatedAt: "2025-05-04T15:34:56.478+00:00"
      },
      requests: [
        {
          contract_id: "6817892000104e73ab7d",
          farmer_id: "68178920001908c8bf3a",
          status: "accepted",
          $id: "68178920000fdfe6133c",
          $createdAt: "2025-05-04T15:34:56.509+00:00",
          $updatedAt: "2025-05-04T15:34:56.509+00:00"
        }
      ]
    },
    "681789200023135d6953": {
      details: {
        buyer_id: "681789200007d68f8bb3",
        status: "fulfilled",
        dynamic_fields: "{\"requirements\": {\"certification\": \"organic\", \"storage_condition\": \"cool dry place\"}}",
        locations: [
          "CHHATTISGARH",
          "Maharashtra"
        ],
        crop_type: "MAIZE",
        quantity: 800,
        price_per_kg: 22,
        advance_payment: 4000,
        delivery_date: "2025-10-31T15:34:56.473+00:00",
        payment_terms: "Cash on delivery",
        $id: "681789200023135d6953",
        $createdAt: "2025-05-04T15:34:56.484+00:00",
        $updatedAt: "2025-05-04T15:34:56.484+00:00"
      },
      requests: [
        {
          contract_id: "681789200023135d6953",
          farmer_id: "68178920001908c8bf3a",
          status: "fulfilled",
          $id: "6817892000077f01f136",
          $createdAt: "2025-05-04T15:34:56.515+00:00",
          $updatedAt: "2025-05-04T15:34:56.515+00:00"
        }
      ]
    },
    "681789200024eb826f11": {
      details: {
        buyer_id: "681789200020953b7072",
        status: "listed",
        dynamic_fields: "{\"requirements\": {\"harvest_window\": \"Oct-Nov\", \"packaging_standard\": \"standard boxes\"}}",
        locations: [
          "CHHATTISGARH",
          "West Bengal"
        ],
        crop_type: "MANGO",
        quantity: 600,
        price_per_kg: 30,
        advance_payment: 3000,
        delivery_date: "2025-10-01T15:34:56.473+00:00",
        payment_terms: "Full payment on delivery",
        $id: "681789200024eb826f11",
        $createdAt: "2025-05-04T15:34:56.490+00:00",
        $updatedAt: "2025-05-04T15:34:56.490+00:00"
      },
      requests: [
        {
          contract_id: "681789200024eb826f11",
          farmer_id: "68178920001908c8bf3a",
          status: "pending",
          $id: "68178920001057340078",
          $createdAt: "2025-05-04T15:34:56.521+00:00",
          $updatedAt: "2025-05-04T15:34:56.521+00:00"
        }
      ]
    },
    "6817892000002c333788": {
      details: {
        buyer_id: "681789200021c64cb1ac",
        status: "listed",
        dynamic_fields: "{\"requirements\": {\"bulk_order_only\": true, \"min_diameter_mm\": 40}}",
        locations: [
          "CHHATTISGARH",
          "Gujarat"
        ],
        crop_type: "ONION",
        quantity: 500,
        price_per_kg: 18,
        advance_payment: 2000,
        delivery_date: "2025-11-05T15:34:56.473+00:00",
        payment_terms: "Advance payment only",
        $id: "6817892000002c333788",
        $createdAt: "2025-05-04T15:34:56.497+00:00",
        $updatedAt: "2025-05-04T15:34:56.497+00:00"
      },
      requests: []
    },
    "6817892000045500bd75": {
      details: {
        buyer_id: "681789200025769524c5",
        status: "listed",
        dynamic_fields: "{\"requirements\": {\"refrigeration_required\": true, \"grade_required\": \"B\"}}",
        locations: [
          "CHHATTISGARH",
          "Karnataka"
        ],
        crop_type: "POTATO",
        quantity: 700,
        price_per_kg: 20,
        advance_payment: 3500,
        delivery_date: "2025-10-21T15:34:56.473+00:00",
        payment_terms: "50% advance, 50% on delivery",
        $id: "6817892000045500bd75",
        $createdAt: "2025-05-04T15:34:56.503+00:00",
        $updatedAt: "2025-05-04T15:34:56.503+00:00"
      },
      requests: []
    }
  };

  // Convert the object to an array for FlatList
  const contractsList = Object.values(contractsData);

  // Filter contracts based on active filter
  const filteredContracts = contractsList.filter(contract => {
    if (activeFilter === "all") return true;
    if (activeFilter === "participated") {
      return contract.requests.length > 0;
    }
    if (activeFilter === activeFilter) {
      return contract.details.status === activeFilter;
    }
    return true;
  });

  const renderRequirements = (dynamicFields: string) => {
    try {
      const parsed = JSON.parse(dynamicFields);
      const requirements = parsed.requirements || {};
      
      return Object.entries(requirements).map(([key, value]) => {
        const formattedKey = key
          .split('_')
          .map(word => word.charAt(0).toUpperCase() + word.slice(1))
          .join(' ');
        
        return (
          <View key={key} style={styles.requirementItem}>
            <Typography variant="caption" style={styles.requirementLabel}>
              {formattedKey}:
            </Typography>
            <Typography variant="caption" style={styles.requirementValue}>
              {String(value)}
            </Typography>
          </View>
        );
      });
    } catch (error) {
      return null;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'accepted':
        return colors.success;
      case 'fulfilled':
        return colors.primary;
      case 'pending':
        return colors.warning;
      case 'listed':
        return colors.info || '#3B82F6';
      default:
        return colors.textSecondary;
    }
  };

  const renderContractItem = ({ item }: { item: ContractWithRequests }) => {
    const { details, requests } = item;
    const deliveryDate = new Date(details.delivery_date);
    const requestStatus = requests.length > 0 ? requests[0].status : 'none';
    
    return (
      <Card variant="elevated" style={styles.contractCard}>
        <View style={styles.contractHeader}>
          <View style={styles.cropTypeContainer}>
            <Typography variant="bodyLarge" style={styles.cropType}>
              {details.crop_type}
            </Typography>
          </View>
          
          <View style={[
            styles.statusBadge, 
            { backgroundColor: getStatusColor(details.status) + '20' }
          ]}>
            <Typography 
              variant="small" 
              style={{ 
                color: getStatusColor(details.status),
                fontWeight: '600' 
              }}
            >
              {details.status.toUpperCase()}
            </Typography>
          </View>
        </View>
        
        <View style={styles.contractDetails}>
          <View style={styles.detailRow}>
            <View style={styles.detailItem}>
              <Package width={16} height={16} stroke={colors.textSecondary} />
              <Typography variant="body" style={styles.detailText}>
                {details.quantity} kg
              </Typography>
            </View>
            
            <View style={styles.detailItem}>
              <DollarSign width={16} height={16} stroke={colors.textSecondary} />
              <Typography variant="body" style={styles.detailText}>
                ₹{details.price_per_kg}/kg
              </Typography>
            </View>
          </View>
          
          <View style={styles.detailRow}>
            <View style={styles.detailItem}>
              <Calendar width={16} height={16} stroke={colors.textSecondary} />
              <Typography variant="body" style={styles.detailText}>
                {deliveryDate.toLocaleDateString('en-US', {
                  year: 'numeric',
                  month: 'short',
                  day: 'numeric'
                })}
              </Typography>
            </View>
            
            <View style={styles.detailItem}>
              <Truck width={16} height={16} stroke={colors.textSecondary} />
              <Typography variant="body" style={styles.detailText}>
                {details.locations.join(', ')}
              </Typography>
            </View>
          </View>
        </View>
        
        <View style={styles.requirements}>
          <Typography variant="caption" style={styles.requirementsTitle}>
            Requirements
          </Typography>
          <View style={styles.requirementsList}>
            {renderRequirements(details.dynamic_fields)}
          </View>
        </View>
        
        <View style={styles.paymentSection}>
          <Typography variant="caption" style={styles.paymentTitle}>
            Payment Terms
          </Typography>
          <Typography variant="body" color="textSecondary">
            {details.payment_terms}
          </Typography>
          <Typography variant="caption" style={styles.advancePayment}>
            Advance payment: ₹{details.advance_payment}
          </Typography>
        </View>
        
        {requests.length > 0 ? (
          <View style={styles.requestStatus}>
            <Typography variant="caption" style={styles.requestStatusTitle}>
              Your application:
            </Typography>
            <View style={[
              styles.requestStatusBadge,
              { backgroundColor: getStatusColor(requestStatus) + '20' }
            ]}>
              {requestStatus === 'pending' && <Clock width={14} height={14} stroke={getStatusColor(requestStatus)} />}
              {requestStatus === 'accepted' && <Check width={14} height={14} stroke={getStatusColor(requestStatus)} />}
              {requestStatus === 'fulfilled' && <Check width={14} height={14} stroke={getStatusColor(requestStatus)} />}
              
              <Typography 
                variant="small" 
                style={{ 
                  color: getStatusColor(requestStatus),
                  fontWeight: '600',
                  marginLeft: 4
                }}
              >
                {requestStatus.toUpperCase()}
              </Typography>
            </View>
          </View>
        ) : (
          <TouchableOpacity 
            style={[styles.applyButton, { backgroundColor: colors.primary }]}
            activeOpacity={0.8}
          >
            <Typography variant="body" style={{ color: 'white', fontWeight: '600' }}>
              Apply for Contract
            </Typography>
          </TouchableOpacity>
        )}
      </Card>
    );
  };

  return (
    <View style={styles.container}>
      <View style={styles.filterContainer}>
        <ScrollableFilter 
          filters={[
            { id: 'all', label: 'All Contracts' },
            { id: 'listed', label: 'Open' },
            { id: 'accepted', label: 'Accepted' },
            { id: 'fulfilled', label: 'Fulfilled' },
            { id: 'participated', label: 'Applied' },
          ]}
          activeFilter={activeFilter}
          onSelectFilter={setActiveFilter}
        />
      </View>
      
      <FlatList
        data={filteredContracts}
        renderItem={renderContractItem}
        keyExtractor={item => item.details.$id}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
        ListEmptyComponent={
          <View style={styles.emptyContainer}>
            <AlertCircle width={32} height={32} stroke={colors.textSecondary} />
            <Typography variant="body" color="textSecondary" style={styles.emptyText}>
              No contracts match your filter
            </Typography>
          </View>
        }
      />
    </View>
  );
};

type FilterItem = {
  id: string;
  label: string;
};

const ScrollableFilter = ({ 
  filters, 
  activeFilter, 
  onSelectFilter 
}: { 
  filters: FilterItem[], 
  activeFilter: string, 
  onSelectFilter: (id: string) => void 
}) => {
  const { colors } = useTheme();
  
  return (
    <ScrollView
      horizontal
      showsHorizontalScrollIndicator={false}
      contentContainerStyle={styles.filterScrollContainer}
    >
      {filters.map(filter => (
        <TouchableOpacity
          key={filter.id}
          style={[
            styles.filterChip,
            activeFilter === filter.id && [
              styles.activeFilterChip,
              { borderColor: colors.primary, backgroundColor: colors.primary + '10' }
            ]
          ]}
          onPress={() => onSelectFilter(filter.id)}
        >
          <Typography 
            variant="small" 
            color={activeFilter === filter.id ? 'primary' : 'textSecondary'} 
            style={activeFilter === filter.id && { fontWeight: '600' }}
          >
            {filter.label}
          </Typography>
        </TouchableOpacity>
      ))}
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  filterContainer: {
    marginBottom: 16,
  },
  filterScrollContainer: {
    paddingHorizontal: 20,
    paddingBottom: 8,
  },
  filterChip: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    marginRight: 12,
    backgroundColor: '#F9FAFB',
  },
  activeFilterChip: {
    borderWidth: 1,
  },
  listContainer: {
    padding: 20,
    paddingBottom: 100,
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
  contractCard: {
    marginBottom: 24,
    padding: 16,
    borderRadius: 12,
  },
  contractHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 16,
  },
  cropTypeContainer: {
    flex: 1,
  },
  cropType: {
    fontSize: 18,
    fontWeight: '700',
  },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  contractDetails: {
    backgroundColor: '#FAFAFA',
    padding: 12,
    borderRadius: 10,
    marginBottom: 16,
  },
  detailRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  detailItem: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  detailText: {
    marginLeft: 8,
  },
  requirements: {
    marginBottom: 16,
  },
  requirementsTitle: {
    fontWeight: '700',
    marginBottom: 8,
  },
  requirementsList: {
    backgroundColor: '#F5F5F5',
    padding: 12,
    borderRadius: 8,
    borderLeftWidth: 3,
    borderLeftColor: '#3B82F6',
  },
  requirementItem: {
    flexDirection: 'row',
    marginBottom: 4,
  },
  requirementLabel: {
    fontWeight: '600',
    marginRight: 8,
  },
  requirementValue: {
    flex: 1,
  },
  paymentSection: {
    marginBottom: 16,
    padding: 12,
    backgroundColor: 'rgba(0, 0, 0, 0.03)',
    borderRadius: 8,
  },
  paymentTitle: {
    fontWeight: '700',
    marginBottom: 8,
  },
  advancePayment: {
    fontWeight: '600',
    marginTop: 8,
  },
  requestStatus: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    backgroundColor: '#F0F9FF',
    padding: 12,
    borderRadius: 8,
  },
  requestStatusTitle: {
    fontWeight: '600',
  },
  requestStatusBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 16,
  },
  applyButton: {
    alignItems: 'center',
    paddingVertical: 12,
    borderRadius: 8,
  },
});

export default MarketContracts;
