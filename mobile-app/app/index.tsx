import { View, Text, StyleSheet, ImageBackground, TouchableOpacity, Dimensions, ScrollView } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useRouter } from "expo-router"
import { useTheme } from "../context/ThemeContext"
import { Feather } from "react-native-feather"
import CircularProgress from "../components/CircularProgress"
import { useFarmStore } from "../store/farmStore"
import { useUserStore } from "../store/userStore"
import {TranslatedText} from "../components/TranslatedText"

const { width, height } = Dimensions.get("window")

export default function OnboardingScreen() {
  const router = useRouter()
  const { colors } = useTheme()
  const { user, logoutUser } = useUserStore()

  const handleGetStarted = () => {
    router.replace("/(tabs)")
  }

  const handleSignOut = () => {
    logoutUser()
    // The _layout useEffect will redirect to login
  }

  return (
    <ImageBackground
      source={{ uri: "https://images.unsplash.com/photo-1535912559317-99a2ae608c53?ixlib=rb-4.0.3" }}
      style={styles.background}
    >
      <SafeAreaView style={styles.container}>
        <ScrollView showsVerticalScrollIndicator={false}>
          <View style={styles.header}>
            <View style={[styles.logoContainer, { backgroundColor: colors.card }]}>
              <Feather width={24} height={24} stroke="#8B5A2B" />
            </View>

            <View style={styles.titleContainer}>
              <Text style={[styles.title, { color: "#8B5A2B" }]}>THE NEW ERA OF</Text>
              <Text style={[styles.titleHighlight, { color: "#4D7C0F" }]}>AGRICULTURE</Text>
            </View>

            <Text style={[styles.subtitle, { color: colors.text }]}>
              Sustainable farming solutions for a better tomorrow.
            </Text>
          </View>

          <View style={styles.metricsContainer}>
            <View style={styles.metricItem}>
              <CircularProgress
                size={80}
                strokeWidth={8}
                progress={0.12}
                progressColor="#4D7C0F"
                label="Growth"
                value="12 cm"
                icon="trending-up"
              />
            </View>

            <View style={styles.metricItem}>
              <CircularProgress
                size={80}
                strokeWidth={8}
                progress={0.75}
                progressColor="#1D4ED8"
                label="Moisture"
                value="75%"
                icon="droplet"
              />
            </View>
          </View>

          {/* <View style={styles.headerActions}>
            <TouchableOpacity 
              onPress={handleSignOut}
              style={[styles.signOutButton, { backgroundColor: colors.error }]}
            >
              <Text style={{ color: colors.primary }}>Sign Out</Text>
            </TouchableOpacity>
          </View> */}

          <TouchableOpacity
            style={[styles.button, { backgroundColor: "rgba(139, 90, 43, 0.8)" }]}
            onPress={handleGetStarted}
          >
            <Text style={styles.buttonText}>Get Started</Text>
          </TouchableOpacity>
        </ScrollView>
      </SafeAreaView>
    </ImageBackground>
  )
}

const styles = StyleSheet.create({
  background: {
    flex: 1,
    width: "100%",
    height: "100%",
  },
  container: {
    flex: 1,
    padding: 24,
    justifyContent: "space-between",
  },
  header: {
    marginTop: height * 0.05,
  },
  logoContainer: {
    width: 40,
    height: 40,
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    marginBottom: 20,
  },
  titleContainer: {
    marginBottom: 12,
  },
  title: {
    fontSize: 28,
    fontWeight: "700",
  },
  titleHighlight: {
    fontSize: 28,
    fontWeight: "800",
  },
  subtitle: {
    fontSize: 16,
    opacity: 0.8,
    maxWidth: "80%",
  },
  metricsContainer: {
    flexDirection: "row",
    justifyContent: "space-around",
    marginVertical: 40,
  },
  metricItem: {
    alignItems: "center",
  },
  button: {
    paddingVertical: 16,
    borderRadius: 30,
    alignItems: "center",
    marginBottom: 40,
    marginTop: 40,
  },
  buttonText: {
    color: "white",
    fontSize: 18,
    fontWeight: "600",
  },
  headerActions: {
    flexDirection: "row",
    alignItems: "center",
  },
  signOutButton: {
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 8,
  },
})
