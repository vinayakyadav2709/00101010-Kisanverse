import type React from "react"
import { View, Text, StyleSheet } from "react-native"
import Svg, { Circle } from "react-native-svg"
import { Droplet, TrendingUp } from "react-native-feather"
import { TranslatedText } from "./TranslatedText"

interface CircularProgressProps {
  size: number
  strokeWidth: number
  progress: number
  progressColor: string
  label: string
  value: string
  icon: "droplet" | "trending-up"
}

const CircularProgress: React.FC<CircularProgressProps> = ({
  size,
  strokeWidth,
  progress,
  progressColor,
  label,
  value,
  icon,
}) => {
  const radius = (size - strokeWidth) / 2
  const circumference = radius * 2 * Math.PI
  const progressValue = circumference - progress * circumference

  return (
    <View style={styles.container}>
      <Svg width={size} height={size}>
        <Circle
          stroke="rgba(255, 255, 255, 0.3)"
          fill="none"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
        />
        <Circle
          stroke={progressColor}
          fill="none"
          cx={size / 2}
          cy={size / 2}
          r={radius}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={progressValue}
          strokeLinecap="round"
        />
      </Svg>
      <View style={styles.iconContainer}>
        {icon === "droplet" ? (
          <Droplet width={18} height={18} stroke="#1D4ED8" />
        ) : (
          <TrendingUp width={18} height={18} stroke="#4D7C0F" />
        )}
      </View>
      <View style={styles.labelContainer}>
        <Text style={styles.label}>{label}</Text>
        <Text style={styles.value}>{value}</Text>
      </View>
    </View>
  )
}

const styles = StyleSheet.create({
  container: {
    alignItems: "center",
    justifyContent: "center",
  },
  iconContainer: {
    position: "absolute",
    backgroundColor: "rgba(255, 255, 255, 0.8)",
    borderRadius: 20,
    padding: 6,
  },
  labelContainer: {
    position: "absolute",
    top: "105%",
    alignItems: "center",
  },
  label: {
    fontSize: 14,
    fontWeight: "600",
    color: "#4B5563",
  },
  value: {
    fontSize: 14,
    fontWeight: "700",
    color: "#1F2937",
  },
})

export default CircularProgress
