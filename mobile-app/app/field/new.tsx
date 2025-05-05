import { View, Text, StyleSheet, TextInput, ScrollView, TouchableOpacity } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../../context/ThemeContext"
import { useFarmStore } from "../../store/farmStore"
import { useUserStore } from "../../store/userStore"
import { useRouter } from "expo-router"
import { ArrowLeft, Camera, Calendar } from "react-native-feather"
import { useState } from "react"
import { TranslatedText } from "../../components/TranslatedText"

export default function NewFieldScreen() {
  const { colors } = useTheme()
  const { addField } = useFarmStore()
  const { user } = useUserStore()
  const router = useRouter()

  const [fieldName, setFieldName] = useState("")
  const [location, setLocation] = useState("")
  const [size, setSize] = useState("")
  const [currentCrop, setCurrentCrop] = useState("")
  const [harvestDate, setHarvestDate] = useState("")

  const handleAddField = () => {
    if (!fieldName || !location || !size || !currentCrop) {
      // Show error
      return
    }

    const newField = {
      id: Date.now().toString(),
      name: fieldName,
      location,
      size: Number.parseFloat(size),
      currentCrop,
      image: "https://images.unsplash.com/photo-1500382017468-9049fed747ef?ixlib=rb-4.0.3",
      soilMoisture: 65,
      soilpH: 6.5,
      lastUpdated: new Date().toISOString(),
      harvestDate: harvestDate || new Date(Date.now() + 90 * 24 * 60 * 60 * 1000).toISOString(),
      expectedYield: "0 kg/ha",
      // Add owner information
      ownerName: user?.name || "Unknown",
      ownerEmail: user?.email || "",
    }

    addField(newField)
    router.back()
  }

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <ScrollView showsVerticalScrollIndicator={false} contentContainerStyle={styles.scrollContent}>
        <View style={styles.header}>
          <TouchableOpacity onPress={() => router.back()} style={styles.backButton}>
            <ArrowLeft width={24} height={24} stroke={colors.text} />
          </TouchableOpacity>
          <Text style={[styles.headerTitle, { color: colors.text }]}>Add New Field</Text>
          <View style={styles.placeholderView} />
        </View>

        <View style={styles.formContainer}>
          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Field Name</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter field name"
              placeholderTextColor={colors.text + "80"}
              value={fieldName}
              onChangeText={setFieldName}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Location</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter location"
              placeholderTextColor={colors.text + "80"}
              value={location}
              onChangeText={setLocation}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Size (hectares)</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter field size"
              placeholderTextColor={colors.text + "80"}
              value={size}
              onChangeText={setSize}
              keyboardType="numeric"
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Current Crop</Text>
            <TextInput
              style={[styles.input, { backgroundColor: colors.card, color: colors.text, borderColor: colors.border }]}
              placeholder="Enter current crop"
              placeholderTextColor={colors.text + "80"}
              value={currentCrop}
              onChangeText={setCurrentCrop}
            />
          </View>

          <View style={styles.inputGroup}>
            <Text style={[styles.label, { color: colors.text }]}>Expected Harvest Date</Text>
            <View
              style={[styles.input, styles.dateInput, { backgroundColor: colors.card, borderColor: colors.border }]}
            >
              <TextInput
                style={[styles.dateTextInput, { color: colors.text }]}
                placeholder="YYYY-MM-DD"
                placeholderTextColor={colors.text + "80"}
                value={harvestDate}
                onChangeText={setHarvestDate}
              />
              <Calendar width={20} height={20} stroke={colors.text} />
            </View>
          </View>

          <TouchableOpacity style={[styles.imageButton, { backgroundColor: colors.card, borderColor: colors.border }]}>
            <Camera width={24} height={24} stroke={colors.text} />
            <Text style={[styles.imageButtonText, { color: colors.text }]}>Capture Field Image</Text>
          </TouchableOpacity>

          <TouchableOpacity style={[styles.submitButton, { backgroundColor: colors.primary }]} onPress={handleAddField}>
            <Text style={styles.submitButtonText}>Add Field</Text>
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
  scrollContent: {
    flexGrow: 1,
    padding: 20,
  },
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24,
  },
  headerTitle: {
    fontSize: 20,
    fontWeight: '600',
  },
  backButton: {
    padding: 4,
  },
  placeholderView: {
    width: 24, // Same as the back button for balanced spacing
  },
  formContainer: {
    flex: 1,
  },
  inputGroup: {
    marginBottom: 16,
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
    paddingHorizontal: 12,
    fontSize: 16,
  },
  dateInput: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
    paddingRight: 12,
  },
  dateTextInput: {
    flex: 1,
    height: 50,
    fontSize: 16,
  },
  imageButton: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    height: 60,
    borderWidth: 1,
    borderRadius: 8,
    marginBottom: 24,
    paddingVertical: 12,
  },
  imageButtonText: {
    fontSize: 16,
    marginLeft: 8,
  },
  submitButton: {
    height: 56,
    borderRadius: 8,
    alignItems: "center",
    justifyContent: "center",
    marginTop: 8,
  },
  submitButtonText: {
    color: "white",
    fontSize: 16,
    fontWeight: "600",
  },
})
