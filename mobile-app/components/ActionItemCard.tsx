import React from "react"
import { View, StyleSheet, TouchableOpacity } from "react-native"
import { useTheme } from "../context/ThemeContext"
import type { ActionItem } from "../store/farmStore"
import { CheckCircle, Calendar } from "react-native-feather"
import { Card } from "./Card"
import { Typography } from "./Typography"

interface ActionItemCardProps {
  item: ActionItem
}

const ActionItemCard: React.FC<ActionItemCardProps> = ({ item }) => {
  const { colors, spacing, radius } = useTheme()

  const getPriorityColor = (priority: "low" | "medium" | "high") => {
    switch (priority) {
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
    <Card
      style={[
        styles.container,
        { borderLeftColor: getPriorityColor(item.priority) }
      ]}
    >
      <View style={styles.header}>
        <Typography variant="bodyLarge" style={{ flex: 1 }}>{item.title}</Typography>
        <View style={[styles.priorityBadge, { backgroundColor: getPriorityColor(item.priority) }]}>
          <Typography variant="small" style={styles.priorityText}>
            {item.priority.charAt(0).toUpperCase() + item.priority.slice(1)}
          </Typography>
        </View>
      </View>

      <Typography variant="caption" color="textSecondary" style={styles.description}>
        {item.description}
      </Typography>

      <View style={styles.footer}>
        <View style={styles.dateContainer}>
          <Calendar width={16} height={16} stroke={colors.textSecondary} />
          <Typography variant="caption" color="textSecondary" style={{ marginLeft: spacing.xs }}>
            Due: {new Date(item.dueDate).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
          </Typography>
        </View>

        <TouchableOpacity 
          style={[
            styles.completeButton, 
            { 
              backgroundColor: colors.primary,
              borderRadius: radius.sm
            }
          ]}
        >
          <CheckCircle width={14} height={14} stroke="white" />
          <Typography variant="small" style={styles.completeText}>Complete</Typography>
        </TouchableOpacity>
      </View>
    </Card>
  )
}

const styles = StyleSheet.create({
  container: {
    marginBottom: 16,
    borderLeftWidth: 4,
  },
  header: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 8,
  },
  priorityBadge: {
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 4,
  },
  priorityText: {
    color: "white",
    fontWeight: "500",
  },
  description: {
    marginBottom: 12,
  },
  footer: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
  },
  dateContainer: {
    flexDirection: "row",
    alignItems: "center",
  },
  completeButton: {
    flexDirection: "row",
    alignItems: "center",
    paddingHorizontal: 12,
    paddingVertical: 6,
  },
  completeText: {
    color: "white",
    marginLeft: 4,
    fontWeight: "500",
  },
})

export default ActionItemCard
