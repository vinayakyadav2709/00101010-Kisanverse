

import { StyleSheet, FlatList } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../context/ThemeContext"
import { useFarmStore } from "../store/farmStore"
import { Stack, useRouter } from "expo-router"
import { ArrowLeft } from "react-native-feather"
import ActionItemCard from "../components/ActionItemCard"
import { TouchableOpacity } from "react-native-gesture-handler"

export default function AllActionsScreen() {
  const { colors } = useTheme()
  const { actionItems } = useFarmStore()
  const router = useRouter()

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <Stack.Screen
        options={{
          headerShown: true,
          headerTitle: "Action Items",
          headerStyle: { backgroundColor: colors.background },
          headerTintColor: colors.text,
          headerLeft: () => (
            <TouchableOpacity onPress={() => router.back()}>
              <ArrowLeft width={24} height={24} stroke={colors.text} />
            </TouchableOpacity>
          ),
        }}
      />

      <FlatList
        data={actionItems}
        keyExtractor={(item) => item.id}
        renderItem={({ item }) => <ActionItemCard item={item} />}
        contentContainerStyle={styles.listContainer}
        showsVerticalScrollIndicator={false}
      />
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  listContainer: {
    padding: 20,
  },
})
