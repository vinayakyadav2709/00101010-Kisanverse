import type React from "react"
import { createContext, useContext, useState, useEffect } from "react"
import { useColorScheme } from "react-native"
import { useFonts } from 'expo-font'

// Define our typography scale
export const typography = {
  headingLarge: {
    fontSize: 28,
    fontWeight: "700" as const,
    lineHeight: 34,
  },
  heading: {
    fontSize: 24,
    fontWeight: "700" as const,
    lineHeight: 32,
  },
  subheading: {
    fontSize: 18,
    fontWeight: "600" as const,
    lineHeight: 24,
  },
  bodyLarge: {
    fontSize: 16,
    fontWeight: "500" as const,
    lineHeight: 22,
  },
  body: {
    fontSize: 16,
    fontWeight: "400" as const,
    lineHeight: 22,
  },
  caption: {
    fontSize: 14,
    fontWeight: "400" as const,
    lineHeight: 18,
  },
  small: {
    fontSize: 12,
    fontWeight: "400" as const,
    lineHeight: 16,
  },
}

// Define spacing scale
export const spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
}

// Define radius scale
export const radius = {
  sm: 4,
  md: 8,
  lg: 16,
  xl: 24,
  round: 9999,
}

type ThemeType = "light" | "dark"

interface ThemeContextType {
  theme: ThemeType
  toggleTheme: () => void
  colors: {
    text: string
    textSecondary: string
    background: string
    backgroundSecondary: string
    primary: string
    secondary: string
    accent: string
    card: string
    border: string
    error: string
    warning: string
    success: string
    info: string
    shadow: string
  }
  typography: typeof typography
  spacing: typeof spacing
  radius: typeof radius
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined)

export const ThemeProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const deviceTheme = useColorScheme() as ThemeType
  const [theme, setTheme] = useState<ThemeType>(deviceTheme || "light")

  // Load fonts
  const [fontsLoaded] = useFonts({
    'Inter-Regular': require('../assets/fonts/Inter-Regular.ttf'),
    'Inter-Medium': require('../assets/fonts/Inter-Medium.ttf'),
    'Inter-SemiBold': require('../assets/fonts/Inter-SemiBold.ttf'),
    'Inter-Bold': require('../assets/fonts/Inter-Bold.ttf'),
  })

  useEffect(() => {
    if (deviceTheme) {
      setTheme(deviceTheme)
    }
  }, [deviceTheme])

  const toggleTheme = () => {
    setTheme((prevTheme) => (prevTheme === "light" ? "dark" : "light"))
  }

  // Enhanced light color palette
  const lightColors = {
    text: "#1A2138",
    textSecondary: "#4A5568",
    background: "#F7FAFC", // Lighter background
    backgroundSecondary: "#EDF2F7",
    primary: "#2563EB", // More vibrant blue
    secondary: "#8B5CF6", // Purple
    accent: "#10B981", // Green
    card: "#FFFFFF",
    border: "#E2E8F0",
    error: "#E53E3E",
    warning: "#F59E0B",
    success: "#10B981", 
    info: "#3B82F6",
    shadow: "rgba(0, 0, 0, 0.1)",
  }

  // Enhanced dark color palette
  const darkColors = {
    text: "#F7FAFC",
    textSecondary: "#A0AEC0",
    background: "#1A202C", // Dark but not pure black
    backgroundSecondary: "#2D3748",
    primary: "#3B82F6", // Slightly lighter blue for dark mode
    secondary: "#9F7AEA", // Lighter purple for dark mode
    accent: "#34D399", // Lighter green for dark mode
    card: "#2D3748",
    border: "#4A5568",
    error: "#F56565",
    warning: "#F6AD55",
    success: "#68D391",
    info: "#63B3ED",
    shadow: "rgba(0, 0, 0, 0.3)",
  }

  const colors = theme === "light" ? lightColors : darkColors

  // Don't render until fonts are loaded
  if (!fontsLoaded) {
    return null
  }

  return (
    <ThemeContext.Provider value={{ theme, toggleTheme, colors, typography, spacing, radius }}>
      {children}
    </ThemeContext.Provider>
  )
}

export const useTheme = () => {
  const context = useContext(ThemeContext)
  if (context === undefined) {
    throw new Error("useTheme must be used within a ThemeProvider")
  }
  return context
}
