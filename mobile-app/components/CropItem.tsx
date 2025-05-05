import React from "react"
import { View, StyleSheet, TouchableOpacity, Text } from "react-native"
import { useTheme } from "../context/ThemeContext"
import type { Crop } from "../store/farmStore"
import { Calendar, DollarSign, AlertTriangle } from "react-native-feather"
import { Card } from "./Card"
import { Typography } from "./Typography"
import { TranslatedText } from "./TranslatedText"

interface CropItemProps {
  crop: Crop
}

const CropItem: React.FC<CropItemProps> = ({ crop }) => {
  const { colors, spacing } = useTheme()

  const getRiskColor = (level: "low" | "medium" | "high") => {
    switch (level) {
      case "low":
        return colors.success
      case "medium":
        return colors.warning
      case "high":
        return colors.error
      default:
        return colors.success
    }
  }

  return (
    <TouchableOpacity activeOpacity={0.7}>
      <Card variant="elevated" style={styles.container}>
        <View style={styles.header}>
          <View style={styles.nameContainer}>
            <Text style={styles.icon}>{crop.icon}</Text>
            <Typography variant="subheading">
              <Text>{crop.name}</Text>
            </Typography>
          </View>
          <View style={[styles.scoreContainer, { backgroundColor: colors.primary }]}>
            <Typography style={styles.score}>{crop.score}</Typography>
          </View>
        </View>

        <View style={styles.divider} />

        <View style={styles.detailsContainer}>
          <View style={styles.detailItem}>
            <Calendar width={16} height={16} stroke={colors.textSecondary} />
            <Typography variant="caption" style={{ marginLeft: spacing.xs }}>
              Plant: {new Date(crop.plantingDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
            </Typography>
          </View>

          <View style={styles.detailItem}>
            <Calendar width={16} height={16} stroke={colors.textSecondary} />
            <Typography variant="caption" style={{ marginLeft: spacing.xs }}>
              Harvest: {new Date(crop.harvestDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
            </Typography>
          </View>

          <View style={styles.detailItem}>
            <DollarSign width={16} height={16} stroke={colors.success} />
            <Typography variant="caption" color="success" style={{ marginLeft: spacing.xs }}>
              Revenue: ${crop.expectedRevenue}
            </Typography>
          </View>

          <View style={styles.detailItem}>
            <AlertTriangle width={16} height={16} stroke={getRiskColor(crop.riskLevel)} />
            <Typography 
              variant="caption" 
              style={{ 
                color: getRiskColor(crop.riskLevel),
                marginLeft: spacing.xs
              }}
            >
              Risk: {crop.riskLevel.charAt(0).toUpperCase() + crop.riskLevel.slice(1)}
            </Typography>
          </View>
        </View>
      </Card>
    </TouchableOpacity>
  )
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  nameContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  icon: {
    fontSize: 24,
    marginRight: 12,
  },
  scoreContainer: {
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: "center",
    alignItems: "center",
  },
  score: {
    color: "white",
    fontWeight: "700",
    fontSize: 16,
  },
  divider: {
    height: 1,
    backgroundColor: 'rgba(0,0,0,0.1)',
    marginVertical: 12,
  },
  detailsContainer: {
    gap: 10,
  },
  detailItem: {
    flexDirection: "row",
    alignItems: "center",
  },
})

export default CropItem
