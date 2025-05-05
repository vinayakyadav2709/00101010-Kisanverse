import { useState, useRef } from "react";
import { 
  View, 
  Text, 
  StyleSheet, 
  TextInput, 
  TouchableOpacity, 
  ScrollView,
  ActivityIndicator,
  Alert,
  Modal,
  FlatList
} from "react-native";
import { TranslatedText } from "../../components/TranslatedText";
import { SafeAreaView } from "react-native-safe-area-context";
import { useTheme } from "../../context/ThemeContext";
import { useUserStore } from "../../store/userStore";
import { useRouter, Link } from "expo-router";
import { ChevronDown, Search, X } from "react-native-feather";

// List of Indian states
const INDIAN_STATES = [
  "ANDHRA PRADESH",
  "TELANGANA",
  "ASSAM",
  "DELHI",
  "GUJARAT",
  "THE DADRA AND NAGAR HAVELI AND DAMAN AND DIU",
  "CHHATTISGARH",
  "BIHAR",
  "HARYANA",
  "MEGHALAYA",
  "KARNATAKA",
  "KERALA",
  "JAMMU AND KASHMIR",
  "LADAKH",
  "JHARKHAND",
  "MADHYA PRADESH",
  "GOA",
  "MAHARASHTRA",
  "HIMACHAL PRADESH",
  "ODISHA",
  "PUNJAB",
  "CHANDIGARH",
  "MANIPUR",
  "MIZORAM",
  "NAGALAND",
  "RAJASTHAN",
  "TAMIL NADU",
  "ARUNACHAL PRADESH",
  "TRIPURA",
  "PUDUCHERRY",
  "UTTAR PRADESH",
  "UTTARAKHAND",
  "WEST BENGAL",
  "ANDAMAN AND NICOBAR ISLANDS",
  "SIKKIM",
  "LAKSHADWEEP"
];

export default function SignupScreen() {
  const { colors } = useTheme();
  const router = useRouter();
  const { registerUser, isLoading, error, clearError } = useUserStore();
  
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [address, setAddress] = useState("");
  const [zipCode, setZipCode] = useState(""); 
  const [role, setRole] = useState("farmer");
  
  // State selection
  const [state, setState] = useState("");
  const [stateModalVisible, setStateModalVisible] = useState(false);
  const [stateSearchQuery, setStateSearchQuery] = useState("");
  const stateSearchRef = useRef<TextInput>(null);

  const validateEmail = (email: string) => {
    return email.match(/^[^\s@]+@[^\s@]+\.[^\s@]+$/);
  };

  const validateZipCode = (zipCode: string) => {
    return /^\d{6}$/.test(zipCode);
  };

  // Filter states based on search query
  const filteredStates = INDIAN_STATES.filter(
    s => s.toLowerCase().includes(stateSearchQuery.toLowerCase())
  );

  const handleSignup = async () => {
    // Clear any previous errors
    clearError();
    
    // Basic validation
    if (!name || !email || !password || !confirmPassword || !address || !zipCode || !state) {
      Alert.alert("Validation Error", "All fields are required");
      return;
    }

    if (!validateEmail(email)) {
      Alert.alert("Validation Error", "Please enter a valid email address");
      return;
    }

    if (password !== confirmPassword) {
      Alert.alert("Validation Error", "Passwords do not match");
      return;
    }

    if (password.length < 6) {
      Alert.alert("Validation Error", "Password should be at least 6 characters");
      return;
    }
    
    if (!validateZipCode(zipCode)) {
      Alert.alert("Validation Error", "Please enter a valid 6-digit PIN code");
      return;
    }

    try {
      await registerUser({
        name,
        email,
        role,
        address: `${address}, ${state}`, // Include state in address
        zipcode: zipCode
      });

      // If no error was thrown, redirect to appropriate home based on role
      if (role === "buyer") {
        router.replace("/(buyer-tabs)");
      } else {
        router.replace("/(tabs)");
      }
    } catch (err) {
      // The error is already handled in the store
    }
  };

  // Show error alert if there's an error
  if (error) {
    Alert.alert("Registration Error", error, [
      { text: "OK", onPress: clearError }
    ]);
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <Text style={[styles.title, { color: colors.text }]}>Create Account</Text>
        <Text style={[styles.subtitle, { color: colors.textSecondary }]}>
          Sign up to start managing your farm or buy crops
        </Text>

        <View style={styles.form}>
          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Full Name</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter your full name"
              placeholderTextColor={colors.textSecondary}
              value={name}
              onChangeText={setName}
              autoCapitalize="words"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Email Address</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter your email"
              placeholderTextColor={colors.textSecondary}
              value={email}
              onChangeText={setEmail}
              keyboardType="email-address"
              autoCapitalize="none"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Password</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter your password"
              placeholderTextColor={colors.textSecondary}
              value={password}
              onChangeText={setPassword}
              secureTextEntry
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Confirm Password</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Confirm your password"
              placeholderTextColor={colors.textSecondary}
              value={confirmPassword}
              onChangeText={setConfirmPassword}
              secureTextEntry
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Address</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter your address"
              placeholderTextColor={colors.textSecondary}
              value={address}
              onChangeText={setAddress}
              multiline
            />
          </View>
          
          {/* State Selection Dropdown */}
          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>State</Text>
            <TouchableOpacity 
              style={[
                styles.input, 
                styles.dropdownButton, 
                { 
                  backgroundColor: colors.card, 
                  borderColor: colors.border,
                  flex: 1
                }
              ]}
              onPress={() => {
                setStateModalVisible(true);
                setStateSearchQuery("");
                // Focus on search input when modal opens
                setTimeout(() => {
                  stateSearchRef.current?.focus();
                }, 100);
              }}
            >
              <Text 
                style={{ 
                  color: state ? colors.text : colors.textSecondary 
                }}
              >
                {state || "Select your state"}
              </Text>
              <Text 
                style={{ 
                  color: state ? colors.text : colors.textSecondary,
                  flex: 1
                }}
              >
                {state || "Select your state"}
              </Text>
              <ChevronDown width={20} height={20} stroke={colors.textSecondary} />
            </TouchableOpacity>
          </View>
          
          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>PIN Code</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter your 6-digit PIN code"
              placeholderTextColor={colors.textSecondary}
              value={zipCode}
              onChangeText={setZipCode}
              keyboardType="numeric"
              maxLength={6}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>I am a</Text>
            <View style={styles.roleSelector}>
              <TouchableOpacity 
                style={[
                  styles.roleOption, 
                  role === 'farmer' && { 
                    backgroundColor: colors.primary + '20',
                    borderColor: colors.primary 
                  }
                ]}
                onPress={() => setRole('farmer')}
              >
                <Text style={[
                  styles.roleText, 
                  { color: role === 'farmer' ? colors.primary : colors.textSecondary }
                ]}>
                  Farmer
                </Text>
              </TouchableOpacity>
              
              <TouchableOpacity 
                style={[
                  styles.roleOption, 
                  role === 'buyer' && { 
                    backgroundColor: colors.primary + '20',
                    borderColor: colors.primary 
                  }
                ]}
                onPress={() => setRole('buyer')}
              >
                <Text style={[
                  styles.roleText, 
                  { color: role === 'buyer' ? colors.primary : colors.textSecondary }
                ]}>
                  Buyer
                </Text>
              </TouchableOpacity>
            </View>
          </View>

          <TouchableOpacity 
            style={[styles.button, { backgroundColor: colors.primary }]} 
            onPress={handleSignup}
            disabled={isLoading}
          >
            {isLoading ? (
              <ActivityIndicator color="white" />
            ) : (
              <Text style={styles.buttonText}>Create Account</Text>
            )}
          </TouchableOpacity>

          <View style={styles.loginTextContainer}>
            <Text style={[styles.loginText, { color: colors.textSecondary }]}>
              {"Already have an account? "}
            </Text>
            <Link href="/auth/login" asChild>
              <TouchableOpacity>
                <Text style={[styles.loginLink, { color: colors.primary }]}>Sign In</Text>
              </TouchableOpacity>
            </Link>
          </View>
        </View>
      </ScrollView>

      {/* State Selection Modal */}
      <Modal
        visible={stateModalVisible}
        transparent={true}
        animationType="slide"
        onRequestClose={() => setStateModalVisible(false)}
      >
        <View style={styles.modalOverlay}>
          <View style={[styles.modalContent, { backgroundColor: colors.card }]}>
            <View style={styles.modalHeader}>
              <Text style={[styles.modalTitle, { color: colors.text }]}>Select State</Text>
              <TouchableOpacity 
                style={styles.closeButton}
                onPress={() => setStateModalVisible(false)}
              >
                <X width={24} height={24} stroke={colors.text} />
              </TouchableOpacity>
            </View>
            
            <View style={[styles.searchContainer, { backgroundColor: colors.backgroundSecondary, borderColor: colors.border }]}>
              <Search width={20} height={20} stroke={colors.textSecondary} style={{ marginRight: 8 }} />
              <TextInput
                ref={stateSearchRef}
                style={[styles.searchInput, { color: colors.text }]}
                placeholder="Search states..."
                placeholderTextColor={colors.textSecondary}
                value={stateSearchQuery}
                onChangeText={setStateSearchQuery}
                autoCapitalize="none"
              />
              {stateSearchQuery.length > 0 && (
                <TouchableOpacity onPress={() => setStateSearchQuery("")}>
                  <X width={18} height={18} stroke={colors.textSecondary} />
                </TouchableOpacity>
              )}
            </View>
            
            <FlatList
              data={filteredStates}
              keyExtractor={(item) => item}
              renderItem={({ item }) => (
                <TouchableOpacity 
                  style={[
                    styles.stateOption,
                    state === item && { backgroundColor: colors.primary + '15' }
                  ]}
                  onPress={() => {
                    setState(item);
                    setStateModalVisible(false);
                  }}
                >
                  <Text style={[
                    styles.stateText, 
                    { color: state === item ? colors.primary : colors.text }
                  ]}>
                    {item}
                  </Text>
                </TouchableOpacity>
              )}
              contentContainerStyle={styles.stateList}
              showsVerticalScrollIndicator={true}
              ListEmptyComponent={
                <View style={styles.emptyState}>
                  <Text style={{ color: colors.textSecondary }}>No states found</Text>
                </View>
              }
            />
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
  scrollContent: {
    flexGrow: 1,
    padding: 20,
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
    marginTop: 20,
    marginBottom: 8,
  },
  subtitle: {
    fontSize: 16,
    marginBottom: 32,
  },
  form: {
    flex: 1,
  },
  inputGroup: {
    marginBottom: 20,
  },
  label: {
    fontSize: 16,
    fontWeight: "500",
    marginBottom: 8,
  },
  input: {
    height: 50,
    borderWidth: 1,
    borderRadius: 8,
    paddingHorizontal: 16,
    fontSize: 16,
  },
  dropdownButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  button: {
    height: 56,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 16,
    marginBottom: 24,
  },
  buttonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
  loginTextContainer: {
    flexDirection: "row",
    justifyContent: "center",
    marginTop: 16,
  },
  loginText: {
    fontSize: 16,
  },
  loginLink: {
    fontSize: 16,
    fontWeight: "600",
  },
  roleSelector: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  roleOption: {
    flex: 1,
    padding: 16,
    borderWidth: 1,
    borderColor: '#E0E0E0',
    borderRadius: 8,
    alignItems: 'center',
    marginHorizontal: 6,
  },
  roleText: {
    fontSize: 16,
    fontWeight: '500',
  },
  // Modal styles
  modalOverlay: {
    flex: 1,
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
    justifyContent: 'flex-end',
  },
  modalContent: {
    borderTopLeftRadius: 16,
    borderTopRightRadius: 16,
    maxHeight: '80%',
    paddingBottom: 20,
  },
  modalHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingTop: 16,
    paddingBottom: 12,
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: '600',
  },
  closeButton: {
    padding: 4,
  },
  searchContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    borderWidth: 1,
    borderRadius: 8,
    marginHorizontal: 20,
    marginBottom: 10,
    height: 44,
  },
  searchInput: {
    flex: 1,
    height: 44,
    fontSize: 16,
  },
  stateList: {
    paddingHorizontal: 20,
  },
  stateOption: {
    paddingVertical: 14,
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  stateText: {
    fontSize: 16,
  },
  emptyState: {
    padding: 20,
    alignItems: 'center',
  }
});
