import { View, Text, StyleSheet, Image, ScrollView, TouchableOpacity } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../../context/ThemeContext"
import { useFarmStore } from "../../store/farmStore"
import { useLocalSearchParams, useRouter } from "expo-router"
import { MapPin, Calendar, Droplet, Activity, ArrowLeft, Edit2 } from "react-native-feather"
import { TranslatedText } from "../../components/TranslatedText"

export default function FieldDetailScreen() {
  const { colors } = useTheme()
  const { fields } = useFarmStore()
  const { id } = useLocalSearchParams()
  const router = useRouter()

  const field = fields.find((f) => f.id === id)

  if (!field) {
    return (
      <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
        <Text style={{ color: colors.text }}>Field not found</Text>
      </SafeAreaView>
    )
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView showsVerticalScrollIndicator={false}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <ArrowLeft width={24} height={24} stroke={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>{field.name}</Text>
          <TouchableOpacity style={styles.editButton}>
            <Edit2 width={24} height={24} stroke={colors.text} />
          </TouchableOpacity>
        </View>
        
        <Image source={{ uri: field.image }} style={styles.fieldImage} />

        <View style={styles.fieldInfoContainer}>
          <View style={styles.locationContainer}>
            <MapPin width={16} height={16} stroke={colors.text} />
            <Text style={[styles.locationText, { color: colors.text }]}>{field.location}</Text>
          </View>

          <View style={[styles.infoCard, { backgroundColor: colors.card, borderColor: colors.border }]}>
            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.text }]}>Size:</Text>
              <Text style={[styles.infoValue, { color: colors.text }]}>{field.size} hectares</Text>
            </View>

            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.text }]}>Current Crop:</Text>
              <Text style={[styles.infoValue, { color: colors.text }]}>{field.currentCrop}</Text>
            </View>

            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.text }]}>Last Updated:</Text>
              <Text style={[styles.infoValue, { color: colors.text }]}>
                {new Date(field.lastUpdated).toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "short",
                  day: "numeric",
                })}
              </Text>
            </View>

            <View style={styles.infoRow}>
              <View style={styles.harvestContainer}>
                <Calendar width={16} height={16} stroke={colors.primary} />
                <Text style={[styles.harvestLabel, { color: colors.text }]}>Harvest Date:</Text>
              </View>
              <Text style={[styles.harvestValue, { color: colors.primary }]}>
                {new Date(field.harvestDate).toLocaleDateString("en-US", {
                  year: "numeric",
                  month: "short",
                  day: "numeric",
                })}
              </Text>
            </View>

            <View style={styles.infoRow}>
              <Text style={[styles.infoLabel, { color: colors.text }]}>Expected Yield:</Text>
              <Text style={[styles.yieldValue, { color: colors.primary }]}>{field.expectedYield}</Text>
            </View>
          </View>

          <Text style={[styles.sectionTitle, { color: colors.text }]}>Soil Conditions</Text>

          <View style={styles.metricsContainer}>
            <View style={[styles.metricCard, { backgroundColor: colors.card, borderColor: colors.border }]}>
              <Droplet width={24} height={24} stroke="#1D4ED8" />
              <Text style={styles.metricValue}>{field.soilMoisture}%</Text>
              <Text style={[styles.metricLabel, { color: colors.text }]}>Soil Moisture</Text>
            </View>

            <View style={[styles.metricCard, { backgroundColor: colors.card, borderColor: colors.border }]}>
              <Activity width={24} height={24} stroke="#10B981" />
              <Text style={styles.metricValue}>pH {field.soilpH}</Text>
              <Text style={[styles.metricLabel, { color: colors.text }]}>Soil pH</Text>
            </View>
          </View>

          <TouchableOpacity style={[styles.actionButton, { backgroundColor: colors.primary }]}>
            <Text style={styles.actionButtonText}>Capture New Soil Image</Text>
          </TouchableOpacity>
        </View>
      </ScrollView>
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 20,
    paddingVertical: 16,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
  },
  backButton: {
    padding: 4,
  },
  editButton: {
    padding: 4,
  },
  fieldImage: {
    width: "100%",
    height: 250,
  },
  fieldInfoContainer: {
    padding: 20,
  },
  locationContainer: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 16,
  },
  locationText: {
    fontSize: 16,
    marginLeft: 6,
  },
  infoCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 24,
    borderWidth: 1,
  },
  infoRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12,
  },
  infoLabel: {
    fontSize: 16,
    fontWeight: "500",
  },
  infoValue: {
    fontSize: 16,
  },
  harvestContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  harvestLabel: {
    fontSize: 16,
    fontWeight: "500",
    marginLeft: 6,
  },
  harvestValue: {
    fontSize: 16,
    fontWeight: "600",
  },
  yieldValue: {
    fontSize: 16,
    fontWeight: "600",
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: "600",
    marginBottom: 16,
  },
  metricsContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 24,
  },
  metricCard: {
    flex: 1,
    alignItems: "center",
    padding: 16,
    borderRadius: 12,
    marginHorizontal: 4,
    borderWidth: 1,
  },
  metricValue: {
    fontSize: 24,
    fontWeight: "700",
    marginVertical: 8,
  },
  metricLabel: {
    fontSize: 14,
  },
  actionButton: {
    paddingVertical: 16,
    borderRadius: 12,
    alignItems: "center",
  },
  actionButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
})
