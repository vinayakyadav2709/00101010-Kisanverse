import type React from "react"
import { View, Text, StyleSheet } from "react-native"
import { useTheme } from "../context/ThemeContext"
import type { Weather } from "../store/farmStore"
import { Thermometer, Droplet, Wind } from "react-native-feather"
import Cloud from "./Cloud"
import { Card } from "./Card"
import { Typography } from "./Typography"

interface WeatherWidgetProps {
  weather: Weather
}

const WeatherWidget: React.FC<WeatherWidgetProps> = ({ weather }) => {
  const { colors, spacing, radius } = useTheme()

  // Calculate the position for the sun indicator based on current time
  const calculateSunPosition = () => {
    const now = new Date()
    const hours = now.getHours()
    const minutes = now.getMinutes()

    // Convert sunrise and sunset times to hours
    const sunriseHours = Number.parseInt(weather.sunrise.split(":")[0])
    const sunsetHours = Number.parseInt(weather.sunset.split(":")[0])

    // Calculate position (0 to 1)
    const dayLength = sunsetHours - sunriseHours
    const currentPosition = (hours - sunriseHours + minutes / 60) / dayLength

    // Clamp between 0 and 1
    return Math.max(0, Math.min(1, currentPosition))
  }

  const sunPosition = calculateSunPosition()

  return (
    <Card variant="elevated" style={styles.container}>
      <View style={styles.temperatureContainer}>
        <View style={styles.cloudIcon}>
          <Cloud width={48} height={48} stroke="white" fill={colors.info} />
        </View>
        <Typography variant="headingLarge" color="text">{weather.temperature}°C</Typography>
      </View>

      <View style={styles.metricsContainer}>
        <View style={styles.metricItem}>
          <Thermometer width={20} height={20} stroke={colors.primary} />
          <Typography variant="bodyLarge">{weather.soilTemperature}°C</Typography>
          <Typography variant="small" color="textSecondary">Soil temp</Typography>
        </View>

        <View style={styles.metricItem}>
          <Droplet width={20} height={20} stroke={colors.info} />
          <Typography variant="bodyLarge">{weather.humidity}%</Typography>
          <Typography variant="small" color="textSecondary">Humidity</Typography>
        </View>

        <View style={styles.metricItem}>
          <Wind width={20} height={20} stroke={colors.secondary} />
          <Typography variant="bodyLarge">{weather.wind} m/s</Typography>
          <Typography variant="small" color="textSecondary">Wind</Typography>
        </View>

        <View style={styles.metricItem}>
          <Droplet width={20} height={20} stroke={colors.info} />
          <Typography variant="bodyLarge">{weather.precipitation} mm</Typography>
          <Typography variant="small" color="textSecondary">Rain</Typography>
        </View>
      </View>

      <View style={styles.sunTracker}>
        <View style={styles.sunPath}>
          <View 
            style={[
              styles.sunIndicator, 
              { 
                backgroundColor: colors.warning,
                left: `${sunPosition * 100}%` 
              }
            ]} 
          />
          <View style={[styles.sunPathLine, { backgroundColor: colors.border }]} />
        </View>
        <View style={styles.sunTimes}>
          <View>
            <Typography variant="caption">{weather.sunrise}</Typography>
            <Typography variant="small" color="textSecondary">Sunrise</Typography>
          </View>
          <View>
            <Typography variant="caption">{weather.sunset}</Typography>
            <Typography variant="small" color="textSecondary">Sunset</Typography>
          </View>
        </View>
      </View>
    </Card>
  )
}

const styles = StyleSheet.create({
  container: {
    marginHorizontal: 20,
    marginVertical: 16,
  },
  temperatureContainer: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 24,
  },
  cloudIcon: {
    marginRight: 16,
  },
  metricsContainer: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginBottom: 24,
  },
  metricItem: {
    alignItems: "center",
  },
  sunTracker: {
    marginTop: 10,
  },
  sunPath: {
    height: 30,
    justifyContent: "center",
    position: "relative",
  },
  sunPathLine: {
    height: 2,
    width: "100%",
  },
  sunIndicator: {
    width: 12,
    height: 12,
    borderRadius: 6,
    position: "absolute",
    top: "50%",
    marginTop: -6,
    zIndex: 1,
  },
  sunTimes: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 8,
  },
})

export default WeatherWidget
