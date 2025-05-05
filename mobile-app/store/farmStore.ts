import { create } from "zustand"

export interface Crop {
  id: string
  name: string
  score: number
  plantingDate: string
  harvestDate: string
  expectedRevenue: number
  riskLevel: "low" | "medium" | "high"
  icon: string
}

export interface Field {
  id: string
  name: string
  location: string
  size: number
  currentCrop: string
  image: string
  soilMoisture: number
  soilpH: number
  lastUpdated: string
  harvestDate: string
  expectedYield: string
}

export interface Weather {
  temperature: number
  soilTemperature: number
  humidity: number
  wind: number
  precipitation: number
  sunrise: string
  sunset: string
  location: string
}

export interface RiskAlert {
  id: string
  type: "pest" | "disease" | "weather"
  severity: "low" | "medium" | "high"
  description: string
  affectedCrops: string[]
  date: string
}

export interface ActionItem {
  id: string
  title: string
  description: string
  dueDate: string
  completed: boolean
  priority: "low" | "medium" | "high"
}

interface FarmState {
  fields: Field[]
  recommendedCrops: Crop[]
  weather: Weather
  riskAlerts: RiskAlert[]
  actionItems: ActionItem[]
  commodities: { id: string; name: string; icon: string }[]

  // Actions
  addField: (field: Field) => void
  updateField: (id: string, updates: Partial<Field>) => void
  completeActionItem: (id: string) => void
}

// Mock data
const mockCrops: Crop[] = [
  {
    id: "1",
    name: "Rice",
    score: 92,
    plantingDate: "2024-06-15",
    harvestDate: "2024-10-30",
    expectedRevenue: 2500,
    riskLevel: "low",
    icon: "🌾",
  },
  {
    id: "2",
    name: "Corn",
    score: 87,
    plantingDate: "2024-05-20",
    harvestDate: "2024-09-15",
    expectedRevenue: 2100,
    riskLevel: "medium",
    icon: "🌽",
  },
  {
    id: "3",
    name: "Potatoes",
    score: 85,
    plantingDate: "2024-04-10",
    harvestDate: "2024-08-20",
    expectedRevenue: 1900,
    riskLevel: "low",
    icon: "🥔",
  },
  {
    id: "4",
    name: "Tomatoes",
    score: 82,
    plantingDate: "2024-05-01",
    harvestDate: "2024-08-15",
    expectedRevenue: 2300,
    riskLevel: "medium",
    icon: "🍅",
  },
  {
    id: "5",
    name: "Soybeans",
    score: 78,
    plantingDate: "2024-06-01",
    harvestDate: "2024-10-15",
    expectedRevenue: 1800,
    riskLevel: "high",
    icon: "🫘",
  },
]

const mockFields: Field[] = [
  {
    id: "1",
    name: "Olive Field",
    location: "Chianti Hills",
    size: 5.2,
    currentCrop: "Olive",
    image: "https://i.ibb.co/LhJkj21q/luke-schlanderer-ZUlepa-Dvs-E-unsplash.jpg",
    soilMoisture: 75,
    soilpH: 6.8,
    lastUpdated: "2024-04-15",
    harvestDate: "2024-12-24",
    expectedYield: "7500 kg/ha",
  },
  {
    id: "2",
    name: "North Wheat Field",
    location: "Chianti Hills",
    size: 8.7,
    currentCrop: "Wheat",
    image: "https://i.ibb.co/tp2QdtkG/combine-in-wheat.jpg",
    soilMoisture: 62,
    soilpH: 7.2,
    lastUpdated: "2024-04-12",
    harvestDate: "2024-07-15",
    expectedYield: "4200 kg/ha",
  },
]

const mockWeather: Weather = {
  temperature: 16,
  soilTemperature: 22,
  humidity: 59,
  wind: 6,
  precipitation: 0,
  sunrise: "5:25 am",
  sunset: "8:04 pm",
  location: "Chianti Hills",
}

const mockRiskAlerts: RiskAlert[] = [
  {
    id: "1",
    type: "pest",
    severity: "high",
    description: "Aphid infestation likely in next 7 days",
    affectedCrops: ["Tomatoes", "Peppers"],
    date: "2024-05-10",
  },
  {
    id: "2",
    type: "weather",
    severity: "medium",
    description: "Potential frost event",
    affectedCrops: ["All crops"],
    date: "2024-05-15",
  },
  {
    id: "3",
    type: "disease",
    severity: "low",
    description: "Early signs of powdery mildew",
    affectedCrops: ["Grapes"],
    date: "2024-05-08",
  },
]

const mockActionItems: ActionItem[] = [
  {
    id: "1",
    title: "Collect soil samples",
    description: "Take soil samples from North field for nutrient analysis",
    dueDate: "2024-05-10",
    completed: false,
    priority: "high",
  },
  {
    id: "2",
    title: "Apply fertilizer",
    description: "Apply nitrogen fertilizer to Olive field",
    dueDate: "2024-05-15",
    completed: false,
    priority: "medium",
  },
  {
    id: "3",
    title: "Irrigation maintenance",
    description: "Check and clean irrigation system",
    dueDate: "2024-05-12",
    completed: false,
    priority: "low",
  },
]

const mockCommodities = [
  { id: "1", name: "Rice", icon: "🌾" },
  { id: "2", name: "Corn", icon: "🌽" },
  { id: "3", name: "Grapes", icon: "🍇" },
  { id: "4", name: "Potato", icon: "🥔" },
  { id: "5", name: "Olive", icon: "🫒" },
  { id: "6", name: "Tomato", icon: "🍅" },
]

export const useFarmStore = create<FarmState>((set) => ({
  fields: mockFields,
  recommendedCrops: mockCrops,
  weather: mockWeather,
  riskAlerts: mockRiskAlerts,
  actionItems: mockActionItems,
  commodities: mockCommodities,

  addField: (field) =>
    set((state) => ({
      fields: [...state.fields, field],
    })),

  updateField: (id, updates) =>
    set((state) => ({
      fields: state.fields.map((field) => (field.id === id ? { ...field, ...updates } : field)),
    })),

  completeActionItem: (id) =>
    set((state) => ({
      actionItems: state.actionItems.map((item) => (item.id === id ? { ...item, completed: true } : item)),
    })),
}))
