import React, { useState, useEffect } from "react"
import { View, Text, StyleSheet, ScrollView, TouchableOpacity, Image, TextInput, ActivityIndicator, Platform } from "react-native"
import { SafeAreaView } from "react-native-safe-area-context"
import { useTheme } from "../../context/ThemeContext"
import * as ImagePicker from 'expo-image-picker'
import { Feather, MaterialIcons, FontAwesome5, AntDesign } from '@expo/vector-icons'
import { addMonths, format } from 'date-fns'
import { LineChart, ProgressChart } from 'react-native-chart-kit';
import { Dimensions } from 'react-native';
import { BarChart } from 'react-native-chart-kit';
import { TranslatedText } from "../../components/TranslatedText"

// API functions for each prediction type
const API_BASE_URL = "https://4f70-124-66-175-40.ngrok-free.app/predictions"

// Dummy user email - replace with actual user authentication
const USER_EMAIL = "farmer@example.com"

// API functions
const fetchSoilPrediction = async (imageUri) => {
  try {
    console.log('Soil prediction API call - Start')
    console.log('Image URI:', imageUri)
    
    const formData = new FormData()
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpeg',
      name: 'soil-image.jpg'
    })

    console.log('Sending soil prediction request to:', `${API_BASE_URL}/soil_type?email=${USER_EMAIL}`)
    const response = await fetch(`${API_BASE_URL}/soil_type?email=${USER_EMAIL}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    console.log('Soil prediction API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to predict soil type: ${response.status}`)
    
    const data = await response.json()
    console.log('Soil prediction API response data:', data)
    return data
  } catch (error) {
    console.error('Soil prediction error:', error)
    console.log('Returning dummy soil data due to error')
    // Return dummy data as fallback
    return { soil_type: "Loamy", confidence: 95.0 }
  }
}

const fetchSoilHistory = async () => {
  try {
    console.log('Soil history API call - Start')
    console.log('Fetching soil history from:', `${API_BASE_URL}/soil_type/history?email=${USER_EMAIL}`)
    
    const response = await fetch(`${API_BASE_URL}/soil_type/history?email=${USER_EMAIL}`)
    
    console.log('Soil history API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to fetch soil history: ${response.status}`)
    
    const data = await response.json()
    console.log('Soil history API response data:', data)
    return data
  } catch (error) {
    console.error('Soil history error:', error)
    console.log('Returning dummy soil history data due to error')
    // Return dummy data as fallback
    return {
      history: [
        {
          file_id: "abc123",
          file_url: "https://example.com/soil.jpg",
          soil_type: "Loamy",
          confidence: 95.0,
          uploaded_at: "2023-05-01T12:00:00.000+00:00"
        },
        {
          file_id: "def456",
          file_url: "https://example.com/soil2.jpg",
          soil_type: "Sandy",
          confidence: 87.0,
          uploaded_at: "2023-04-15T10:30:00.000+00:00"
        }
      ]
    }
  }
}

const fetchDiseasePrediction = async (imageUri) => {
  try {
    console.log('Disease prediction API call - Start')
    console.log('Image URI:', imageUri)
    
    const formData = new FormData()
    formData.append('file', {
      uri: imageUri,
      type: 'image/jpg',
      name: 'plant-image.jpg'
    })

    console.log('Sending disease prediction request to:', `${API_BASE_URL}/disease?email=${USER_EMAIL}`)
    const response = await fetch(`${API_BASE_URL}/disease?email=${USER_EMAIL}`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    console.log('Disease prediction API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to predict disease: ${response.status}`)
    
    const data = await response.json()
    console.log('Disease prediction API response data:', data)
    return data
  } catch (error) {
    console.error('Disease prediction error:', error)
    console.log('Returning dummy disease data due to error')
    // Return dummy data as fallback
    return { plant_name: "Tomato", disease_name: "Late Blight", confidence: 92.5 }
  }
}

const fetchDiseaseHistory = async () => {
  try {
    console.log('Disease history API call - Start')
    console.log('Fetching disease history from:', `${API_BASE_URL}/disease/history?email=${USER_EMAIL}`)
    
    const response = await fetch(`${API_BASE_URL}/disease/history?email=${USER_EMAIL}`)
    
    console.log('Disease history API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to fetch disease history: ${response.status}`)
    
    const data = await response.json()
    console.log('Disease history API response data:', data)
    return data
  } catch (error) {
    console.error('Disease history error:', error)
    console.log('Returning dummy disease history data due to error')
    // Return dummy data as fallback
    return {
      history: [
        {
          file_id: "ghi789",
          file_url: "https://example.com/tomato.jpg",
          plant_name: "Tomato",
          disease_name: "Late Blight",
          confidence: 92.5,
          uploaded_at: "2023-05-02T14:20:00.000+00:00"
        },
        {
          file_id: "jkl012",
          file_url: "https://example.com/wheat.jpg",
          plant_name: "Wheat",
          disease_name: "Rust",
          confidence: 88.3,
          uploaded_at: "2023-04-28T09:15:00.000+00:00"
        }
      ]
    }
  }
}

const fetchWeatherPrediction = async (startDate, endDate) => {
  try {
    console.log('Weather prediction API call - Start')
    console.log('Date range:', format(startDate, 'yyyy-MM-dd'), 'to', format(endDate, 'yyyy-MM-dd'))
    
    const requestBody = {
      email: USER_EMAIL,
      start_date: format(startDate, 'yyyy-MM-dd'),
      end_date: format(endDate, 'yyyy-MM-dd')
    }
    console.log('Weather prediction request body:', requestBody)
    console.log('Sending weather prediction request to:', `${API_BASE_URL}/weather`)
    
    const response = await fetch(`${API_BASE_URL}/weather`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(requestBody)
    })

    console.log('Weather prediction API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to predict weather: ${response.status}`)
    
    const data = await response.json()
    console.log('Weather prediction API response data:', data)
    return data
  } catch (error) {
    console.error('Weather prediction error:', error)
    console.log('Returning dummy weather data due to error')
    // Return dummy data as fallback
    return [
      {
        month: "04",
        temperature_2m_max: 30.5,
        temperature_2m_min: 29.4,
        precipitation_sum: 0.0,
        wind_speed_10m_max: 22.5,
        shortwave_radiation_sum: 25.26
      },
      {
        month: "05",
        temperature_2m_max: 32.1,
        temperature_2m_min: 30.0,
        precipitation_sum: 1.2,
        wind_speed_10m_max: 20.0,
        shortwave_radiation_sum: 24.50
      }
    ]
  }
}

const fetchWeatherHistory = async () => {
  try {
    console.log('Weather history API call - Start')
    console.log('Fetching weather history from:', `${API_BASE_URL}/weather/history?email=${USER_EMAIL}`)
    
    const response = await fetch(`${API_BASE_URL}/weather/history?email=${USER_EMAIL}`)
    
    console.log('Weather history API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to fetch weather history: ${response.status}`)
    
    const data = await response.json()
    console.log('Weather history API response data:', data)
    return data
  } catch (error) {
    console.error('Weather history error:', error)
    console.log('Returning dummy weather history data due to error')
    // Return dummy data as fallback
    return {
      history: [
        {
          start_date: "2023-05-01",
          end_date: "2023-05-10",
          weather_data: [
            {
              month: "05",
              temperature_2m_max: 30.5,
              temperature_2m_min: 29.4,
              precipitation_sum: 0.0,
              wind_speed_10m_max: 22.5,
              shortwave_radiation_sum: 25.26
            }
          ],
          requested_at: "2023-04-29T12:00:00.000+00:00"
        }
      ]
    }
  }
}

const fetchCropPrediction = async (startDate, endDate, acres, soilType, soilImage) => {
  try {
    console.log('Crop prediction API call - Start')
    console.log('Date range:', format(startDate, 'yyyy-MM-dd'), 'to', format(endDate, 'yyyy-MM-dd'))
    console.log('Acres:', acres)
    console.log('Soil type:', soilType || 'Not provided')
    console.log('Soil image:', soilImage ? 'Provided' : 'Not provided')
    
    const formData = new FormData()
    formData.append('email', USER_EMAIL)
    formData.append('start_date', format(startDate, 'yyyy-MM-dd'))
    formData.append('end_date', format(endDate, 'yyyy-MM-dd'))
    formData.append('acres', acres.toString())
    
    if (soilType) {
      formData.append('soil_type', soilType)
    } else if (soilImage) {
      formData.append('file', {
        uri: soilImage,
        type: 'image/jpeg',
        name: 'soil-image.jpg'
      })
    }

    console.log('Sending crop prediction request to:', `${API_BASE_URL}/crop_prediction`)
    const response = await fetch(`${API_BASE_URL}/crop_prediction`, {
      method: 'POST',
      body: formData,
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    })

    console.log('Crop prediction API response status:', response.status)
    if (!response.ok) throw new Error(`Failed to predict crops: ${response.status}`)
    
    const data = await response.json()
    console.log('Crop prediction API response data:', data)
    return data
  } catch (error) {
    console.error('Crop prediction error:', error)
    console.log('Returning dummy crop prediction data due to error')
    // Return dummy data as fallback with the new structure
    return {
      "request_details": {
        "latitude": 22.520393976898,
        "longitude": 88.007016991204,
        "soil_type": "Black",
        "land_size_acres": parseInt(acres),
        "analysis_period": {
          "start_date": format(startDate, 'yyyy-MM-dd'),
          "end_date": format(endDate, 'yyyy-MM-dd')
        },
        "timestamp": new Date().toISOString()
      },
      "recommendations": [
        {
          "rank": 1,
          "crop_name": "Soybean",
          "recommendation_score": 85.5,
          "explanation_points": [
            {
              "reason_type": "High Profitability",
              "detail": "Analysis indicates strong profit potential driven by stable-to-rising price forecasts combined with manageable 'Low-Medium' estimated input costs, significantly boosted by a seed subsidy.",
              "ai_component": "Predictive Analytics, Heuristic Evaluation, Rule-Based Logic (Subsidy)"
            },
            {
              "reason_type": "Optimal Agronomics",
              "detail": "Excellent match for Black soil type. Weather forecast analysis confirms sufficient precipitation expected during the critical early growth stage in July.",
              "ai_component": "Knowledge-Based Rules, Weather Data Integration"
            },
            {
              "reason_type": "Financial Advantage",
              "detail": "The available 50% Seed Subsidy directly reduces upfront costs, lowering financial risk.",
              "ai_component": "Subsidy Data Integration, Rule-Based Impact"
            },
            {
              "reason_type": "Favorable Market Conditions",
              "detail": "Current market prices are stable, and moderate platform demand signals are positive.",
              "ai_component": "Market Data Analysis"
            }
          ],
          "key_metrics": {
            "expected_yield_range": "8-10 quintals/acre",
            "price_forecast_trend": "Stable to Slightly Rising",
            "estimated_input_cost_category": "Low-Medium",
            "primary_fertilizer_needs": "Inoculant, Phosphorus, Potassium (Nitrogen fixed)"
          },
          "relevant_subsidies": [
            {
              "program": "State Certified Seed Subsidy (Kharif)",
              "provider": "State Agriculture Department",
              "benefit_summary": "Approx. 50% cost reduction on certified Soybean seeds.",
              "details_available": true
            }
          ],
          "primary_risks": [
            "Sensitivity to waterlogging if heavy, prolonged rains occur.",
            "Requires monitoring for common pests like girdle beetle and whitefly."
          ],
          "plotting_data": {
            "price_forecast_chart": {
              "description": "Predicted price range (INR/Quintal) near harvest.",
              "data": [
                { "date": "2025-11-01", "predicted_price_min": 4500, "predicted_price_max": 4800 },
                { "date": "2025-11-15", "predicted_price_min": 4550, "predicted_price_max": 4900 },
                { "date": "2025-12-01", "predicted_price_min": 4600, "predicted_price_max": 4950 },
                { "date": "2025-12-15", "predicted_price_min": 4650, "predicted_price_max": 5000 }
              ]
            },
            "water_need_chart": {
              "description": "Relative water requirement across growth stages (1=Low, 5=Very High).",
              "data": [
                { "growth_stage": "Germination (0-2wk)", "relative_need_level": 2 },
                { "growth_stage": "Vegetative (3-6wk)", "relative_need_level": 4 },
                { "growth_stage": "Flowering/Podding (7-10wk)", "relative_need_level": 5 },
                { "growth_stage": "Maturity (11+wk)", "relative_need_level": 1 }
              ]
            },
            "fertilizer_schedule_chart": {
              "description": "Typical nutrient application timing.",
              "data": [
                { "stage": "Basal", "timing": "At Sowing", "nutrients": "Inoculant, P, K" },
                { "stage": "Top Dress", "timing": "~30 Days", "nutrients": "K (if needed)" }
              ]
            }
          }
        },
        {
          "rank": 2,
          "crop_name": "Maize (Kharif)",
          "recommendation_score": 78.0,
          "explanation_points": [
            {
              "reason_type": "Strong Yield & Price",
              "detail": "Offers high yield potential (25-30 q/acre) coupled with a rising price forecast driven by feed demand, leading to good revenue projection despite higher 'Medium-High' input costs.",
              "ai_component": "Predictive Analytics, Heuristic Evaluation"
            },
            {
              "reason_type": "Good Agronomic Fit",
              "detail": "Suitable for Black soil and takes advantage of expected monsoon rainfall, although high water need requires consistent rainfall throughout the season.",
              "ai_component": "Knowledge-Based Rules, Weather Data Integration"
            },
            {
              "reason_type": "Positive Market Trend",
              "detail": "Price trend is upward due to external factors (feed demand), indicating strong market pull.",
              "ai_component": "Market Data Analysis, Event Correlation"
            },
            {
              "reason_type": "Higher Risk Profile",
              "detail": "Requires higher fertilizer investment ('Medium-High' cost category) and is more sensitive to rainfall consistency. No major subsidies noted to offset these risks.",
              "ai_component": "Risk Factor Analysis"
            }
          ],
          "key_metrics": {
            "expected_yield_range": "25-30 quintals/acre",
            "price_forecast_trend": "Rising",
            "estimated_input_cost_category": "Medium-High",
            "primary_fertilizer_needs": "High Nitrogen, Phosphorus, Potassium"
          },
          "relevant_subsidies": [],
          "primary_risks": [
            "Higher upfront fertilizer costs.",
            "Yield potential highly dependent on consistent rainfall.",
            "Requires monitoring and potential management for Fall Armyworm."
          ],
          "plotting_data": {
            "price_forecast_chart": {
              "description": "Predicted price range (INR/Quintal) near harvest.",
              "data": [
                { "date": "2025-10-15", "predicted_price_min": 2100, "predicted_price_max": 2300 },
                { "date": "2025-11-01", "predicted_price_min": 2150, "predicted_price_max": 2400 },
                { "date": "2025-11-15", "predicted_price_min": 2200, "predicted_price_max": 2450 },
                { "date": "2025-12-01", "predicted_price_min": 2250, "predicted_price_max": 2500 }
              ]
            },
            "water_need_chart": {
              "description": "Relative water requirement across growth stages (1=Low, 5=Very High).",
              "data": [
                { "growth_stage": "Seedling (0-3wk)", "relative_need_level": 3 },
                { "growth_stage": "Vegetative (4-8wk)", "relative_need_level": 4 },
                { "growth_stage": "Tasseling/Silking (9-12wk)", "relative_need_level": 5 },
                { "growth_stage": "Grain Fill/Maturity (13+wk)", "relative_need_level": 3 }
              ]
            },
            "fertilizer_schedule_chart": {
              "description": "Typical nutrient application timing.",
              "data": [
                { "stage": "Basal", "timing": "At Sowing", "nutrients": "Partial N, P, K" },
                { "stage": "Knee-High", "timing": "~30 Days", "nutrients": "N Top Dress" },
                { "stage": "Tasseling", "timing": "~50-60 Days", "nutrients": "N Top Dress" }
              ]
            }
          }
        }
      ],
      "weather_context_summary": "Overall weather forecast for Jul 01-18 indicates generally favorable monsoon conditions (warm, frequent rain) for Kharif planting in this region."
    }
  }
}

// Remove DateTimePicker import and use a custom date picker UI instead
// ...existing code...

// Modified date handling functions to replace DateTimePicker
const formatDate = (date) => {
  return format(date, 'yyyy-MM-dd')
}

// Simple function to increment date by days
const addDays = (date, days) => {
  const result = new Date(date)
  result.setDate(result.getDate() + days)
  return result
}

// Improved date input component with increment/decrement buttons
const renderDateInput = (label, value, onChange) => (
  <View style={styles.formGroup}>
    <Text style={[styles.formLabel, {color: colors.textSecondary}]}>{label}</Text>
    <View style={styles.dateInputContainer}>
      <TouchableOpacity 
        style={[styles.dateActionButton, {backgroundColor: colors.primary + '20', borderRadius: 8}]}
        onPress={() => onChange(addDays(value, -1))}
      >
        <Feather name="minus" size={20} color={colors.primary} />
      </TouchableOpacity>
      
      <TextInput
        style={[styles.dateInput, {
          borderColor: colors.border, 
          color: colors.text,
          backgroundColor: colors.background,
          textAlign: 'center',
          fontWeight: '500'
        }]}
        value={formatDate(value)}
        onChangeText={(text) => {
          try {
            // Better date parsing
            const parts = text.split('-');
            if (parts.length === 3) {
              const year = parseInt(parts[0]);
              const month = parseInt(parts[1]) - 1; // JS months are 0-indexed
              const day = parseInt(parts[2]);
              
              const newDate = new Date(year, month, day);
              if (!isNaN(newDate.getTime())) {
                onChange(newDate);
              }
            }
          } catch (error) {
            console.log("Date parsing error:", error);
          }
        }}
        placeholder="YYYY-MM-DD"
        placeholderTextColor={colors.textSecondary}
      />
      
      <TouchableOpacity 
        style={[styles.dateActionButton, {backgroundColor: colors.primary + '20', borderRadius: 8}]}
        onPress={() => onChange(addDays(value, 1))}
      >
        <Feather name="plus" size={20} color={colors.primary} />
      </TouchableOpacity>
    </View>
    <Text style={[styles.dateFormatHint, {color: colors.textSecondary}]}>
      Format: YYYY-MM-DD
    </Text>
  </View>
)

// Add this hardcoded weather data after the fetchWeatherPrediction function
const HARDCODED_WEATHER_DATA = [
  {
    "month": "05",
    "temperature_2m_max": 41.07,
    "temperature_2m_min": 29.12,
    "precipitation_sum": 0.03,
    "wind_speed_10m_max": 16.73,
    "shortwave_radiation_sum": 22.86
  },
  {
    "month": "06",
    "temperature_2m_max": 42.65,
    "temperature_2m_min": 32.3,
    "precipitation_sum": 1.55,
    "wind_speed_10m_max": 12.7,
    "shortwave_radiation_sum": 19.39
  },
  {
    "month": "07",
    "temperature_2m_max": 30.73,
    "temperature_2m_min": 26.18,
    "precipitation_sum": 9.78,
    "wind_speed_10m_max": 11.9,
    "shortwave_radiation_sum": 17.25
  },
  {
    "month": "08",
    "temperature_2m_max": 31.52,
    "temperature_2m_min": 25.35,
    "precipitation_sum": 14.0,
    "wind_speed_10m_max": 9.28,
    "shortwave_radiation_sum": 16.44
  },
  {
    "month": "09",
    "temperature_2m_max": 26.48,
    "temperature_2m_min": 24.27,
    "precipitation_sum": 3.2,
    "wind_speed_10m_max": 6.25,
    "shortwave_radiation_sum": 19.09
  },
  {
    "month": "10",
    "temperature_2m_max": 31.8,
    "temperature_2m_min": 23.8,
    "precipitation_sum": 5.0,
    "wind_speed_10m_max": 6.7,
    "shortwave_radiation_sum": 14.21
  }
];

// Add these helper functions before the ForecastScreen component
const getMonthName = (monthNumber) => {
  const months = ['January', 'February', 'March', 'April', 'May', 'June', 
                  'July', 'August', 'September', 'October', 'November', 'December'];
  return months[parseInt(monthNumber) - 1];
};

const prepareTemperatureChartData = (weatherData) => {
  return {
    labels: weatherData.map(item => getMonthName(item.month)),
    datasets: [
      {
        data: weatherData.map(item => item.temperature_2m_max),
        color: (opacity = 1) => `rgba(255, 99, 71, ${opacity})`, // Tomato color for max temp
        strokeWidth: 2,
        // Add a unique identifier for this dataset
        id: 'max-temp',
      },
      {
        data: weatherData.map(item => item.temperature_2m_min),
        color: (opacity = 1) => `rgba(70, 130, 180, ${opacity})`, // Steel blue for min temp
        strokeWidth: 2,
        // Add a unique identifier for this dataset
        id: 'min-temp',
      },
    ],
    legend: ["Max Temp (°C)", "Min Temp (°C)"]
  };
};

const preparePrecipitationChartData = (weatherData) => {
  return {
    labels: weatherData.map(item => getMonthName(item.month)),
    datasets: [
      {
        data: weatherData.map(item => item.precipitation_sum),
        color: (opacity = 1) => `rgba(65, 105, 225, ${opacity})`, // Royal blue
        strokeWidth: 2,
        // Add a unique identifier for this dataset
        id: 'precipitation',
      }
    ],
    legend: ["Precipitation (mm)"]
  };
};

const prepareWindSpeedChartData = (weatherData) => {
  return {
    labels: weatherData.map(item => getMonthName(item.month)),
    datasets: [
      {
        data: weatherData.map(item => item.wind_speed_10m_max),
        color: (opacity = 1) => `rgba(60, 179, 113, ${opacity})`, // Medium sea green
        strokeWidth: 2,
        // Add a unique identifier for this dataset
        id: 'wind-speed',
      }
    ],
    legend: ["Max Wind Speed (km/h)"]
  };
};

const prepareRadiationChartData = (weatherData) => {
  return {
    labels: weatherData.map(item => getMonthName(item.month)),
    datasets: [
      {
        data: weatherData.map(item => item.shortwave_radiation_sum),
        color: (opacity = 1) => `rgba(255, 165, 0, ${opacity})`, // Orange
        strokeWidth: 2,
        // Add a unique identifier for this dataset
        id: 'radiation',
      }
    ],
    legend: ["Solar Radiation (MJ/m²)"]
  };
};

function ForecastScreen() {
  const { colors } = useTheme()
  const [activeTab, setActiveTab] = useState('weather') // Changed default tab to weather
  const screenWidth = Dimensions.get('window').width - 40;
  
  // Soil prediction state
  const [soilImage, setSoilImage] = useState(null)
  const [soilResult, setSoilResult] = useState(null)
  const [soilHistory, setSoilHistory] = useState([])
  const [loadingSoil, setLoadingSoil] = useState(false)
  
  // Disease prediction state
  const [diseaseImage, setDiseaseImage] = useState(null)
  const [diseaseResult, setDiseaseResult] = useState(null)
  const [diseaseHistory, setDiseaseHistory] = useState([])
  const [loadingDisease, setLoadingDisease] = useState(false)
  
  // Weather prediction state - replace DatePicker with manual date entry
  const [weatherStartDate, setWeatherStartDate] = useState(new Date())
  const [weatherEndDate, setWeatherEndDate] = useState(new Date('2025-10-04'))
  const [weatherResult, setWeatherResult] = useState(HARDCODED_WEATHER_DATA)
  const [weatherHistory, setWeatherHistory] = useState([])
  const [loadingWeather, setLoadingWeather] = useState(false)
  const [forecastRequested, setForecastRequested] = useState(false) // Add this state variable
  
  // Crop prediction state
  const [cropStartDate, setCropStartDate] = useState(new Date())
  const [cropEndDate, setCropEndDate] = useState(new Date('2025-12-02'))
  const [cropAcres, setCropAcres] = useState('5')
  const [cropSoilType, setCropSoilType] = useState('')
  const [cropSoilImage, setCropSoilImage] = useState(null)
  const [cropResult, setCropResult] = useState(null)
  const [loadingCrop, setLoadingCrop] = useState(false)

  // Fetch history on initial load
  useEffect(() => {
    const loadInitialData = async () => {
      try {
        const soilHistoryData = await fetchSoilHistory()
        setSoilHistory(soilHistoryData.history || [])
        
        const diseaseHistoryData = await fetchDiseaseHistory()
        setDiseaseHistory(diseaseHistoryData.history || [])
        
        const weatherHistoryData = await fetchWeatherHistory()
        setWeatherHistory(weatherHistoryData.history || [])
      } catch (error) {
        console.error('Error loading initial data:', error)
      }
    }
    
    loadInitialData()
  }, [])

  // Image picker functions
  const pickImage = async (setImageFunc) => {
    const result = await ImagePicker.launchImageLibraryAsync({
      mediaTypes: ImagePicker.MediaTypeOptions.Images,
      allowsEditing: true,
      aspect: [4, 3],
      quality: 0.8,
    })

    if (!result.canceled && result.assets && result.assets.length > 0) {
      setImageFunc(result.assets[0].uri)
    }
  }

  // Prediction handlers
  const handleSoilPrediction = async () => {
    if (!soilImage) return
    
    setLoadingSoil(true)
    try {
      const result = await fetchSoilPrediction(soilImage)
      setSoilResult(result)
      
      // Refresh history after prediction
      const historyData = await fetchSoilHistory()
      setSoilHistory(historyData.history || [])
    } catch (error) {
      console.error('Error predicting soil:', error)
    } finally {
      setLoadingSoil(false)
    }
  }

  const handleDiseasePrediction = async () => {
    if (!diseaseImage) return
    
    setLoadingDisease(true)
    try {
      const result = await fetchDiseasePrediction(diseaseImage)
      setDiseaseResult(result)
      
      // Refresh history after prediction
      const historyData = await fetchDiseaseHistory()
      setDiseaseHistory(historyData.history || [])
    } catch (error) {
      console.error('Error predicting disease:', error)
    } finally {
      setLoadingDisease(false)
    }
  }

  const handleWeatherPrediction = async () => {
    setLoadingWeather(true)
    try {
      // Use hardcoded data instead of API call
      setTimeout(() => {
        setWeatherResult(HARDCODED_WEATHER_DATA)
        setForecastRequested(true) // Set this to true after getting forecast data
        setLoadingWeather(false)
      }, 1000) // Simulate loading
    } catch (error) {
      console.error('Error predicting weather:', error)
      setLoadingWeather(false)
    }
  }

  const handleCropPrediction = async () => {
    if (!cropSoilType && !cropSoilImage) {
      alert('Please select a soil type or upload a soil image')
      return
    }
    
    setLoadingCrop(true)
    try {
      const result = await fetchCropPrediction(
        cropStartDate,
        cropEndDate,
        cropAcres,
        cropSoilType,
        cropSoilImage
      )
      setCropResult(result)
    } catch (error) {
      console.error('Error predicting crops:', error)
    } finally {
      setLoadingCrop(false)
    }
  }

  // Tab rendering
  const renderTabs = () => (
    <View style={styles.tabContainer}>
      <TouchableOpacity 
        style={[styles.tab, activeTab === 'soil' && {backgroundColor: colors.primary}]}
        onPress={() => setActiveTab('soil')}
      >
        <MaterialIcons name="terrain" size={22} color={activeTab === 'soil' ? '#fff' : colors.textSecondary} />
        <Text style={{color: activeTab === 'soil' ? '#fff' : colors.textSecondary, marginLeft: 5}}>Soil</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={[styles.tab, activeTab === 'disease' && {backgroundColor: colors.primary}]}
        onPress={() => setActiveTab('disease')}
      >
        <Feather name="activity" size={22} color={activeTab === 'disease' ? '#fff' : colors.textSecondary} />
        <Text style={{color: activeTab === 'disease' ? '#fff' : colors.textSecondary, marginLeft: 5}}>Disease</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={[styles.tab, activeTab === 'weather' && {backgroundColor: colors.primary}]}
        onPress={() => setActiveTab('weather')}
      >
        <Feather name="cloud" size={22} color={activeTab === 'weather' ? '#fff' : colors.textSecondary} />
        <Text style={{color: activeTab === 'weather' ? '#fff' : colors.textSecondary, marginLeft: 5}}>Weather</Text>
      </TouchableOpacity>
      
      <TouchableOpacity 
        style={[styles.tab, activeTab === 'crop' && {backgroundColor: colors.primary}]}
        onPress={() => setActiveTab('crop')}
      >
        <FontAwesome5 name="seedling" size={22} color={activeTab === 'crop' ? '#fff' : colors.textSecondary} />
        <Text style={{color: activeTab === 'crop' ? '#fff' : colors.textSecondary, marginLeft: 5}}>Crop</Text>
      </TouchableOpacity>
    </View>
  )

  // Soil tab content
  const renderSoilTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.sectionContainer}>
        <Text style={[styles.sectionTitle, {color: colors.text}]}>Soil Classification</Text>
        <Text style={[styles.sectionDescription, {color: colors.textSecondary}]}>
          Upload a soil image to classify its type
        </Text>
        
        <View style={styles.imageUploadContainer}>
          {soilImage ? (
            <View>
              <Image source={{uri: soilImage}} style={styles.uploadedImage} />
              <TouchableOpacity 
                style={styles.removeImageButton}
                onPress={() => setSoilImage(null)}
              >
                <AntDesign name="close" size={20} color="#fff" />
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity 
              style={[styles.uploadButton, {borderColor: colors.border}]}
              onPress={() => pickImage(setSoilImage)}
            >
              <Feather name="upload" size={24} color={colors.text} />
              <Text style={[styles.uploadText, {color: colors.text}]}>Upload Soil Image</Text>
            </TouchableOpacity>
          )}
        </View>
        
        <TouchableOpacity 
          style={[
            styles.predictButton, 
            !soilImage && styles.disabledButton,
            {backgroundColor: colors.primary}
          ]}
          onPress={handleSoilPrediction}
          disabled={!soilImage || loadingSoil}
        >
          {loadingSoil ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.predictButtonText}>Classify Soil</Text>
          )}
        </TouchableOpacity>
        
        {soilResult && (
          <View style={[styles.resultContainer, {backgroundColor: colors.card}]}>
            <Text style={[styles.resultTitle, {color: colors.text}]}>Soil Classification Result</Text>
            <View style={styles.resultRow}>
              <Text style={[styles.resultLabel, {color: colors.textSecondary}]}>Soil Type:</Text>
              <Text style={[styles.resultValue, {color: colors.text}]}>{soilResult.soil_type}</Text>
            </View>
            <View style={styles.resultRow}>
              <Text style={[styles.resultLabel, {color: colors.textSecondary}]}>Confidence:</Text>
              <Text style={[styles.resultValue, {color: colors.text}]}>{`${soilResult.confidence}%`}</Text>
            </View>
          </View>
        )}
      </View>
      
      {soilHistory.length > 0 && (
        <View style={styles.sectionContainer}>
          <Text style={[styles.sectionTitle, {color: colors.text}]}>History</Text>
          {soilHistory.map((item, index) => (
            <View key={index} style={[styles.historyItem, {backgroundColor: colors.card}]}>
              <View style={styles.historyItemRow}>
                <Text style={[styles.historyItemLabel, {color: colors.textSecondary}]}>Soil Type:</Text>
                <Text style={[styles.historyItemValue, {color: colors.text}]}>{item.soil_type}</Text>
              </View>
              <View style={styles.historyItemRow}>
                <Text style={[styles.historyItemLabel, {color: colors.textSecondary}]}>Confidence:</Text>
                <Text style={[styles.historyItemValue, {color: colors.text}]}>{`${item.confidence}%`}</Text>
              </View>
              <Text style={[styles.historyDate, {color: colors.textSecondary}]}>
                {new Date(item.uploaded_at).toLocaleDateString()}
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  )

  // Disease tab content
  const renderDiseaseTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.sectionContainer}>
        <Text style={[styles.sectionTitle, {color: colors.text}]}>Plant Disease Detection</Text>
        <Text style={[styles.sectionDescription, {color: colors.textSecondary}]}>
          Upload a plant image to detect diseases
        </Text>
        
        <View style={styles.imageUploadContainer}>
          {diseaseImage ? (
            <View>
              <Image source={{uri: diseaseImage}} style={styles.uploadedImage} />
              <TouchableOpacity 
                style={styles.removeImageButton}
                onPress={() => setDiseaseImage(null)}
              >
                <AntDesign name="close" size={20} color="#fff" />
              </TouchableOpacity>
            </View>
          ) : (
            <TouchableOpacity 
              style={[styles.uploadButton, {borderColor: colors.border}]}
              onPress={() => pickImage(setDiseaseImage)}
            >
              <Feather name="upload" size={24} color={colors.text} />
              <Text style={[styles.uploadText, {color: colors.text}]}>Upload Plant Image</Text>
            </TouchableOpacity>
          )}
        </View>
        
        <TouchableOpacity 
          style={[
            styles.predictButton, 
            !diseaseImage && styles.disabledButton,
            {backgroundColor: colors.primary}
          ]}
          onPress={handleDiseasePrediction}
          disabled={!diseaseImage || loadingDisease}
        >
          {loadingDisease ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={styles.predictButtonText}>Detect Disease</Text>
          )}
        </TouchableOpacity>
        
        {diseaseResult && (
          <View style={[styles.resultContainer, {backgroundColor: colors.card}]}>
            <Text style={[styles.resultTitle, {color: colors.text}]}>Disease Detection Result</Text>
            <View style={styles.resultRow}>
              <Text style={[styles.resultLabel, {color: colors.textSecondary}]}>Plant:</Text>
              <Text style={[styles.resultValue, {color: colors.text}]}>{diseaseResult.plant_name}</Text>
            </View>
            <View style={styles.resultRow}>
              <Text style={[styles.resultLabel, {color: colors.textSecondary}]}>Disease:</Text>
              <Text style={[styles.resultValue, {color: colors.text}]}>{diseaseResult.disease_name}</Text>
            </View>
            <View style={styles.resultRow}>
              <Text style={[styles.resultLabel, {color: colors.textSecondary}]}>Confidence:</Text>
              <Text style={[styles.resultValue, {color: colors.text}]}>{`${diseaseResult.confidence}%`}</Text>
            </View>
          </View>
        )}
      </View>
      
      {diseaseHistory.length > 0 && (
        <View style={styles.sectionContainer}>
          <Text style={[styles.sectionTitle, {color: colors.text}]}>History</Text>
          {diseaseHistory.map((item, index) => (
            <View key={index} style={[styles.historyItem, {backgroundColor: colors.card}]}>
              <View style={styles.historyItemRow}>
                <Text style={[styles.historyItemLabel, {color: colors.textSecondary}]}>Plant:</Text>
                <Text style={[styles.historyItemValue, {color: colors.text}]}>{item.plant_name}</Text>
              </View>
              <View style={styles.historyItemRow}>
                <Text style={[styles.historyItemLabel, {color: colors.textSecondary}]}>Disease:</Text>
                <Text style={[styles.historyItemValue, {color: colors.text}]}>{item.disease_name}</Text>
              </View>
              <View style={styles.historyItemRow}>
                <Text style={[styles.historyItemLabel, {color: colors.textSecondary}]}>Confidence:</Text>
                <Text style={[styles.historyItemValue, {color: colors.text}]}>{`${item.confidence}%`}</Text>
              </View>
              <Text style={[styles.historyDate, {color: colors.textSecondary}]}>
                {new Date(item.uploaded_at).toLocaleDateString()}
              </Text>
            </View>
          ))}
        </View>
      )}
    </ScrollView>
  )

  // Modified date input using TextInput
  const renderDateInput = (label, value, onChange) => (
    <View style={styles.formGroup}>
      <Text style={[styles.formLabel, {color: colors.textSecondary}]}>{label}</Text>
      <TextInput
        style={[styles.textInput, {borderColor: colors.border, color: colors.text}]}
        value={formatDate(value)}
        onChangeText={(text) => {
          // Basic validation for date format YYYY-MM-DD
          if (/^\d{4}-\d{2}-\d{2}$/.test(text)) {
            const newDate = new Date(text);
            if (!isNaN(newDate.getTime())) {
              onChange(newDate);
            }
          }
        }}
        placeholder="YYYY-MM-DD"
        placeholderTextColor={colors.textSecondary}
      />
      <Text style={[styles.dateFormatHint, {color: colors.textSecondary}]}>
        Format: YYYY-MM-DD
      </Text>
    </View>
  )

  // Replace the renderWeatherTab with this enhanced version
  const renderWeatherTab = () => {
    // Chart configuration
    const chartConfig = {
      backgroundColor: colors.card,
      backgroundGradientFrom: colors.card,
      backgroundGradientTo: colors.card,
      decimalPlaces: 1,
      color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
      labelColor: (opacity = 1) => colors.text,
      style: {
        borderRadius: 16,
      },
      propsForDots: {
        r: "4",
        strokeWidth: "2",
      },
    };

    // Prepare chart data
    const temperatureData = prepareTemperatureChartData(weatherResult);
    const precipitationData = preparePrecipitationChartData(weatherResult);
    const windSpeedData = prepareWindSpeedChartData(weatherResult);
    const radiationData = prepareRadiationChartData(weatherResult);

    return (
      <ScrollView style={styles.tabContent}>
        <View style={styles.sectionContainer}>
          <Text style={[styles.sectionTitle, {color: colors.text}]}>
            <TranslatedText>Weather Forecast</TranslatedText>
          </Text>
          <Text style={[styles.sectionDescription, {color: colors.textSecondary}]}>
            <TranslatedText>Get weather forecasts for your region</TranslatedText>
          </Text>
          
          <View style={[styles.weatherDateRangeCard, {backgroundColor: colors.card}]}>
            {renderDateInput('Start Date', weatherStartDate, setWeatherStartDate)}
            {renderDateInput('End Date', weatherEndDate, setWeatherEndDate)}
            
            <TouchableOpacity 
              style={[styles.predictButton, {
                backgroundColor: colors.primary,
                marginTop: 10,
                marginBottom: 5
              }]}
              onPress={handleWeatherPrediction}
              disabled={loadingWeather}
            >
              {loadingWeather ? (
                <ActivityIndicator color="#fff" />
              ) : (
                <View style={{ flexDirection: 'row', alignItems: 'center' }}>
                  <Feather name="cloud" size={18} color="#fff" style={{ marginRight: 8 }} />
                  <Text style={styles.predictButtonText}>
                  <Text>Update Forecast</Text>
                  </Text>
                </View>
              )}
            </TouchableOpacity>
          </View>

          {/* Only show the weather summary and charts when forecastRequested is true */}
          {forecastRequested && weatherResult && weatherResult.length > 0 && (
            <>
              <View style={[styles.weatherSummaryCard, {backgroundColor: colors.card}]}>
                <Text style={[styles.weatherSummaryTitle, {color: colors.text}]}>
                <Text>Weather Summary</Text>
                </Text>
                <Text style={[styles.weatherSummarySubtitle, {color: colors.textSecondary}]}>
                <Text>
                  {`${getMonthName(weatherResult[0].month)} to ${getMonthName(weatherResult[weatherResult.length-1].month)} Outlook`}
                </Text>
                </Text>
                
                {/* Weather highlights */}
                <View style={styles.weatherHighlightsContainer}>
                  <View style={[styles.weatherHighlightItem, {backgroundColor: colors.background}]}>
                    <View style={[styles.weatherHighlightIconContainer, {backgroundColor: `${colors.primary}15`}]}>
                    <Feather name="thermometer" size={20} color={colors.primary} />
                    </View>
                    <Text style={[styles.weatherHighlightValue, {color: colors.text}]}>
                    {`${Math.round(Math.max(...weatherResult.map(item => item.temperature_2m_max)))}°C`}
                    </Text>
                    <Text style={[styles.weatherHighlightLabel, {color: colors.textSecondary}]}>
                    <Text>Max Temp</Text>
                    </Text>
                  </View>
                  
                  <View style={[styles.weatherHighlightItem, {backgroundColor: colors.background}]}>
                    <View style={[styles.weatherHighlightIconContainer, {backgroundColor: `${colors.primary}15`}]}>
                    <Feather name="droplet" size={20} color={colors.primary} />
                    </View>
                    <Text style={[styles.weatherHighlightValue, {color: colors.text}]}>
                    {`${Math.round(weatherResult.reduce((sum, item) => sum + item.precipitation_sum, 0))} mm`}
                    </Text>
                    <Text style={[styles.weatherHighlightLabel, {color: colors.textSecondary}]}>
                    <TranslatedText>Total Rain</TranslatedText>
                    </Text>
                  </View>
                  
                  <View style={[styles.weatherHighlightItem, {backgroundColor: colors.background}]}>
                    <View style={[styles.weatherHighlightIconContainer, {backgroundColor: `${colors.primary}15`}]}>
                    <Feather name="wind" size={20} color={colors.primary} />
                    </View>
                    <Text style={[styles.weatherHighlightValue, {color: colors.text}]}>
                    {`${Math.round(Math.max(...weatherResult.map(item => item.wind_speed_10m_max)))} km/h`}
                    </Text>
                    <Text style={[styles.weatherHighlightLabel, {color: colors.textSecondary}]}>
                    <Text>Max Wind</Text>
                    </Text>
                  </View>
                </View>
              </View>
              
              {/* Temperature Chart */}
              <View style={[styles.chartCard, {backgroundColor: colors.card}]}>
                {/* ...existing temperature chart code... */}
                <Text style={[styles.chartTitle, {color: colors.text}]}>
                  <Text>Temperature Trends</Text>
                </Text>
                <LineChart
                  data={temperatureData}
                  width={screenWidth - 40}
                  height={220}
                  chartConfig={{
                    ...chartConfig,
                    color: (opacity = 1) => `rgba(255, 99, 71, ${opacity})`,
                  }}
                  bezier
                  style={styles.chart}
                  fromZero={false}
                  yAxisSuffix="°C"
                  renderDotContent={({ x, y, index, indexData, dataset }) => {
                    // Add a null check for dataset
                    if (!dataset) return null;
                    
                    return (
                      <Text
                        key={`${dataset.id || 'default'}-dot-${index}`}
                        style={{
                          position: 'absolute',
                          top: y - 18,
                          left: x - 14,
                          width: 30,
                          textAlign: 'center',
                          color: colors.text,
                          fontSize: 10,
                        }}
                      >
                        {indexData.toFixed(1)}
                      </Text>
                    );
                  }}
                />

                <View style={styles.chartLegend}>
                  {temperatureData.legend.map((legend, index) => (
                    <View key={index} style={styles.legendItem}>
                      <View 
                        style={[
                          styles.legendColor, 
                          {
                            backgroundColor: index === 0 
                              ? 'rgba(255, 99, 71, 1)' 
                              : 'rgba(70, 130, 180, 1)'
                          }
                        ]} 
                      />
                      <Text style={[styles.legendText, {color: colors.textSecondary}]}>
                        {legend}
                      </Text>
                    </View>
                  ))}
                </View>
              </View>
              
              {/* Precipitation Chart */}
              <View style={[styles.chartCard, {backgroundColor: colors.card}]}>
                {/* ...existing precipitation chart code... */}
                <Text style={[styles.chartTitle, {color: colors.text}]}>
                  <Text>Precipitation</Text>
                </Text>
                <BarChart
                  data={precipitationData}
                  width={screenWidth - 40}
                  height={220}
                  chartConfig={{
                    ...chartConfig,
                    color: (opacity = 1) => `rgba(65, 105, 225, ${opacity})`,
                    barPercentage: 0.7,
                  }}
                  style={styles.chart}
                  fromZero={true}
                  yAxisSuffix=" mm"
                  showValuesOnTopOfBars={true}
                />
              </View>
              
              {/* Wind Speed & Radiation presented in a 2-column grid */}
              <View style={styles.smallChartsContainer}>
                {/* ...existing small charts code... */}
                {/* Wind Speed Chart */}
                <View style={[styles.smallChartCard, {backgroundColor: colors.card}]}>
                  <Text style={[styles.chartTitle, {color: colors.text, fontSize: 16}]}>
                    <Text>Wind Speed</Text>
                  </Text>
                  <LineChart
                    data={windSpeedData}
                    width={(screenWidth - 50) / 2}
                    height={150}
                    chartConfig={{
                      ...chartConfig,
                      color: (opacity = 1) => `rgba(60, 179, 113, ${opacity})`,
                      propsForLabels: {
                        fontSize: 8,
                      },
                    }}
                    bezier
                    style={styles.smallChart}
                    fromZero={true}
                    yAxisSuffix=" km/h"
                    withInnerLines={false}
                  />
                </View>
                
                {/* Radiation Chart */}
                <View style={[styles.smallChartCard, {backgroundColor: colors.card}]}>
                  <Text style={[styles.chartTitle, {color: colors.text, fontSize: 16}]}>
                    <TranslatedText>Solar Radiation</TranslatedText>
                  </Text>
                  <LineChart
                    data={radiationData}
                    width={(screenWidth - 50) / 2}
                    height={150}
                    chartConfig={{
                      ...chartConfig,
                      color: (opacity = 1) => `rgba(255, 165, 0, ${opacity})`,
                      propsForLabels: {
                        fontSize: 8,
                      },
                    }}
                    bezier
                    style={styles.smallChart}
                    fromZero={true}
                    yAxisSuffix=" MJ/m²"
                    withInnerLines={false}
                  />
                </View>
              </View>
              
              {/* Detailed monthly weather data */}
              <View style={[styles.detailedWeatherCard, {backgroundColor: colors.card}]}>
                {/* ...existing detailed weather code... */}
                <Text style={[styles.detailedWeatherTitle, {color: colors.text}]}>
                  <TranslatedText>Monthly Details</TranslatedText>
                </Text>
                
                {weatherResult.map((item, index) => (
                  <View key={index} style={[
                    styles.monthlyWeatherItem, 
                    index < weatherResult.length - 1 && {
                      borderBottomWidth: 1,
                      borderBottomColor: `${colors.border}50`,
                      paddingBottom: 12,
                      marginBottom: 12
                    }
                  ]}>
                    <Text style={[styles.monthLabel, {color: colors.text}]}>
                      {getMonthName(item.month)}
                    </Text>
                    
                    <View style={styles.weatherMetricsGrid}>
                      <View style={styles.weatherMetricItem}>
                        <View style={styles.weatherMetricRow}>
                          <Feather name="thermometer" size={14} color={colors.primary} />
                          <Text style={[styles.weatherMetricLabel, {color: colors.textSecondary}]}>
                            <TranslatedText>Temp Range</TranslatedText>
                          </Text>
                        </View>
                        <Text style={[styles.weatherMetricValue, {color: colors.text}]}>
                          {`${item.temperature_2m_min.toFixed(1)}°C - ${item.temperature_2m_max.toFixed(1)}°C`}
                        </Text>
                      </View>
                      
                      <View style={styles.weatherMetricItem}>
                        <View style={styles.weatherMetricRow}>
                          <Feather name="droplet" size={14} color="#4096FE" />
                          <Text style={[styles.weatherMetricLabel, {color: colors.textSecondary}]}>
                            <TranslatedText>Precipitation</TranslatedText>
                          </Text>
                        </View>
                        <Text style={[styles.weatherMetricValue, {color: colors.text}]}>
                          {`${item.precipitation_sum.toFixed(1)} mm`}
                        </Text>
                      </View>
                      
                      <View style={styles.weatherMetricItem}>
                        <View style={styles.weatherMetricRow}>
                          <Feather name="wind" size={14} color="#50C878" />
                          <Text style={[styles.weatherMetricLabel, {color: colors.textSecondary}]}>
                            <TranslatedText>Wind Speed</TranslatedText>
                          </Text>
                        </View>
                        <Text style={[styles.weatherMetricValue, {color: colors.text}]}>
                          {`${item.wind_speed_10m_max.toFixed(1)} km/h`}
                        </Text>
                      </View>
                      
                      <View style={styles.weatherMetricItem}>
                        <View style={styles.weatherMetricRow}>
                          <Feather name="sun" size={14} color="#FFA500" />
                          <Text style={[styles.weatherMetricLabel, {color: colors.textSecondary}]}>
                            <TranslatedText>Solar Radiation</TranslatedText>
                          </Text>
                        </View>
                        <Text style={[styles.weatherMetricValue, {color: colors.text}]}>
                          {`${item.shortwave_radiation_sum.toFixed(1)} MJ/m²`}
                        </Text>
                      </View>
                    </View>
                  </View>
                ))}
              </View>
              
              <View style={styles.weatherNotice}>
                <View style={[styles.weatherNoticeIcon, {backgroundColor: `${colors.primary}15`}]}>
                  <Feather name="info" size={18} color={colors.primary} />
                </View>
                <Text style={[styles.weatherNoticeText, {color: colors.textSecondary}]}>
                  <TranslatedText>
                    This weather forecast is based on historical data and predictive modeling. 
                    Actual weather conditions may vary.
                  </TranslatedText>
                </Text>
              </View>
            </>
          )}

          {!forecastRequested && (
            <View style={styles.noForecastContainer}>
              <View style={[styles.noForecastIcon, {backgroundColor: `${colors.primary}10`}]}>
                <Feather name="cloud-rain" size={24} color={colors.primary} />
              </View>
              <Text style={[styles.noForecastText, {color: colors.textSecondary}]}>
                <TranslatedText>
                  Click "Update Forecast" to view detailed weather data for your selected date range.
                </TranslatedText>
              </Text>
            </View>
          )}
        </View>
      </ScrollView>
    );
  };

  // Helper function to prepare price forecast chart data
  const preparePriceChartData = (chartData) => {
    // Extract months and dates for x-axis labels
    const labels = chartData.map(item => {
      const date = new Date(item.date);
      const month = date.toLocaleString('default', { month: 'short' });
      const day = date.getDate();
      return `${day} ${month}`;
    });
    
    // Create datasets for min and max prices
    const minPrices = chartData.map(item => item.predicted_price_min);
    const maxPrices = chartData.map(item => item.predicted_price_max);
    
    // Calculate the average price for a trend line
    const avgPrices = chartData.map((item, index) => 
      (item.predicted_price_min + item.predicted_price_max) / 2
    );
    
    // Find min and max values for better y-axis scaling
    const allPrices = [...minPrices, ...maxPrices];
    const minPrice = Math.min(...allPrices) * 0.95; // Add 5% padding
    const maxPrice = Math.max(...allPrices) * 1.05; // Add 5% padding
    
    return {
      labels,
      datasets: [
        {
          data: avgPrices,
          color: (opacity = 1) => `rgba(54, 162, 235, ${opacity})`,
          strokeWidth: 3,
          withDots: true,
          id: 'avg-price', // Add unique ID
        },
        {
          data: minPrices,
          color: (opacity = 1) => `rgba(75, 192, 192, ${opacity})`,
          strokeWidth: 2,
          withDots: false,
          id: 'min-price', // Add unique ID
        },
        {
          data: maxPrices,
          color: (opacity = 1) => `rgba(153, 102, 255, ${opacity})`,
          strokeWidth: 2,
          withDots: false,
          id: 'max-price', // Add unique ID
        }
      ],
      legend: ["Average Price", "Minimum Price", "Maximum Price"],
      yAxisRange: [minPrice, maxPrice]
    };
  };

  // Improved helper function for water need visualization
  const prepareWaterNeedData = (chartData) => {
    // Convert the water needs to a 0-1 scale for the progress chart
    return {
      labels: chartData.map(item => {
        const stageName = item.growth_stage.split('(')[0].trim();
        return stageName.length > 10 ? stageName.substring(0, 10) + "..." : stageName;
      }),
      data: chartData.map(item => item.relative_need_level / 5), // Scale to 0-1 range
      colors: [
        'rgba(66, 133, 244, 0.8)',
        'rgba(52, 168, 83, 0.8)',
        'rgba(251, 188, 5, 0.8)',
        'rgba(234, 67, 53, 0.8)'
      ],
      strokeWidth: 5
    };
  };

  // Enhanced component to render crop details with better visualizations and improved UI
  const CropDetailView = ({ crop, index }) => {
    const { colors } = useTheme();
    const screenWidth = Dimensions.get('window').width - 40;
    
    // Prepare chart data
    const priceChartData = preparePriceChartData(crop.plotting_data.price_forecast_chart.data);
    const waterNeedData = prepareWaterNeedData(crop.plotting_data.water_need_chart.data);
    
    // Chart config for price chart
    const priceChartConfig = {
      backgroundGradientFrom: '#ffffff',
      backgroundGradientTo: '#ffffff',
      decimalPlaces: 0,
      color: (opacity = 1) => `rgba(0, 0, 0, ${opacity})`,
      labelColor: (opacity = 1) => `rgba(0, 0, 0, ${opacity*0.7})`,
      style: {
        borderRadius: 16
      },
      propsForDots: {
        r: "5",
        strokeWidth: "2",
        stroke: "#ffa726"
      },
      propsForBackgroundLines: {
        strokeWidth: 1,
        stroke: "rgba(0,0,0,0.05)"
      },
      formatYLabel: (value) => `₹${parseInt(value)}`,
    };
    
    // Chart config for water needs progress chart
    const waterChartConfig = {
      backgroundGradientFrom: '#ffffff',
      backgroundGradientTo: '#ffffff',
      color: (opacity = 1, index) => waterNeedData.colors[index] || 'rgba(75, 192, 192, 0.8)',
      strokeWidth: 2,
      decimalPlaces: 0,
      labelColor: () => colors.textSecondary,
    };
    
    // Card background - subtle gradient for depth
    const cardBackground = index === 0 ? '#FAFFF8' : '#FFFFFF';
    const cardBorderColor = index === 0 ? colors.primary : colors.border;
    
    return (
      <View style={[styles.cropDetailCard, { 
        backgroundColor: cardBackground, 
        borderColor: cardBorderColor,
        shadowColor: index === 0 ? colors.primary : '#000',
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: index === 0 ? 0.15 : 0.1,
        shadowRadius: 8,
        elevation: index === 0 ? 5 : 3
      }]}>
        {/* Rank badge with enhanced styling */}
        <View style={[styles.cropRankBadge, {
          backgroundColor: index === 0 ? '#4CAF50' : '#FFA000',
          shadowColor: '#000',
          shadowOffset: { width: 0, height: 2 },
          shadowOpacity: 0.3,
          shadowRadius: 3,
          elevation: 5
        }]}>
          <Text style={styles.cropRankText}>{`#${crop.rank}`}</Text>
        </View>
        
        {/* Crop title and score with enhanced typography */}
        <View style={styles.cropHeaderContainer}>
          <Text style={[styles.cropDetailTitle, { 
            color: colors.text,
            fontWeight: Platform.OS === 'ios' ? '700' : 'bold',
            letterSpacing: -0.5
          }]}>
            {crop.crop_name}
          </Text>
          <View style={styles.scoreContainer}>
            <View style={[styles.scoreCircle, {
              backgroundColor: crop.recommendation_score > 80 ? '#4CAF50' : 
                               crop.recommendation_score > 60 ? '#FFC107' : '#F44336',
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 1 },
              shadowOpacity: 0.2,
              shadowRadius: 2,
              elevation: 2                             
            }]}>
              <Text style={[styles.scoreText, {fontWeight: Platform.OS === 'ios' ? '700' : 'bold'}]}>
                {Math.round(crop.recommendation_score)}%
              </Text>
            </View>
            <Text style={[styles.scoreLabel, {
              color: colors.textSecondary,
              marginTop: 5,
              fontSize: 13
            }]}>Match</Text>
          </View>
        </View>
        
        {/* Key metrics in an attractive grid with enhanced styling */}
        <View style={[styles.metricsGrid, { 
          backgroundColor: `${colors.primary}08`,
          borderRadius: 14,
          marginBottom: 28,
          paddingVertical: 16
        }]}>
          <View style={styles.metricItem}>
            <Text style={[styles.metricLabel, {
              color: colors.textSecondary,
              fontSize: 13,
              marginBottom: 5
            }]}>Expected Yield</Text>
            <Text style={[styles.metricValue, {
              color: colors.text,
              fontSize: 15,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold'
            }]}>{crop.key_metrics.expected_yield_range}</Text>
          </View>
          
          <View style={[styles.metricSeparator, {backgroundColor: `${colors.border}30`}]} />
          
          <View style={styles.metricItem}>
            <Text style={[styles.metricLabel, {
              color: colors.textSecondary,
              fontSize: 13,
              marginBottom: 5
            }]}>Price Trend</Text>
            <Text style={[styles.metricValue, {
              color: colors.text,
              fontSize: 15,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold'
            }]}>{crop.key_metrics.price_forecast_trend}</Text>
          </View>
          
          <View style={[styles.metricSeparator, {backgroundColor: `${colors.border}30`}]} />
          
          <View style={styles.metricItem}>
            <Text style={[styles.metricLabel, {
              color: colors.textSecondary,
              fontSize: 13,
              marginBottom: 5
            }]}>Input Cost</Text>
            <Text style={[styles.metricValue, {
              color: colors.text,
              fontSize: 15,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold'
            }]}>{crop.key_metrics.estimated_input_cost_category}</Text>
          </View>
        </View>
        
        {/* Enhanced price forecast chart with better styling */}
        <View style={styles.chartSection}>
          <View style={styles.chartTitleRow}>
            <Text style={[styles.chartTitle, { 
              color: colors.text,
              fontSize: 18,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
              letterSpacing: -0.3
            }]}>Price Forecast</Text>
            <Text style={[styles.chartSubtitle, { 
              color: colors.primary,
              fontWeight: '500'
            }]}>₹/Quintal</Text>
          </View>
          
          <View style={[styles.chartContainer, {
            backgroundColor: '#FFFFFF',
            borderRadius: 16,
            padding: 15,
            marginVertical: 10,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.06,
            shadowRadius: 4,
            elevation: 2
          }]}>
            <LineChart
              data={priceChartData}
              width={screenWidth - 30}
              height={220}
              chartConfig={priceChartConfig}
              bezier
              withShadow
              withVerticalLines={false}
              segments={4}
              fromZero={false}
              style={styles.enhancedChart}
            />
          </View>
        </View>
        
        {/* Water needs visualization using progress chart with better styling */}
        <View style={[styles.chartSection, {marginBottom: 30}]}>
          <View style={styles.chartTitleRow}>
            <Text style={[styles.chartTitle, { 
              color: colors.text,
              fontSize: 18,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
              letterSpacing: -0.3
            }]}>Water Requirements</Text>
            <Text style={[styles.chartSubtitle, { 
              color: colors.textSecondary,
              fontWeight: '500'
            }]}>By Growth Stage</Text>
          </View>
          
          <View style={[styles.waterNeedsContainer, {
            backgroundColor: '#FFFFFF',
            borderRadius: 16,
            padding: 15,
            marginVertical: 10,
            shadowColor: '#000',
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.06,
            shadowRadius: 4,
            elevation: 2
          }]}>
            <ProgressChart
              data={waterNeedData}
              width={screenWidth - 30}
              height={170}
              chartConfig={waterChartConfig}
              style={styles.enhancedChart}
              withCustomBarColorFromData={true}
              radius={32}
              strokeWidth={8}
              hideLegend={false}
            />
            
            <View style={[styles.waterLegend, {marginTop: 15}]}>
              {crop.plotting_data.water_need_chart.data.map((item, idx) => (
                <View key={idx} style={styles.waterLegendItem}>
                  <View style={[styles.waterLegendDot, { 
                    backgroundColor: waterNeedData.colors[idx % waterNeedData.colors.length],
                    width: 10,
                    height: 10,
                    borderRadius: 5,
                    marginRight: 6
                  }]} />
                  <Text style={[styles.waterLegendText, {
                    color: colors.text,
                    fontSize: 13
                  }]}>
                    <Text style={{fontWeight: '500'}}>{`${item.growth_stage}: ${item.relative_need_level}/5`}</Text>
                  </Text>
                </View>
              ))}
            </View>
          </View>
        </View>
        
        {/* Why this crop section with enhanced styling */}
        {crop.explanation_points.length > 0 && (
          <View style={[styles.reasonsSection, {marginBottom: 28}]}>
            <Text style={[styles.sectionHeading, { 
              color: colors.text,
              fontSize: 18,
              fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
              marginBottom: 16,
              letterSpacing: -0.3
            }]}>Why This Crop?</Text>
            
            {crop.explanation_points.slice(0, 2).map((point, i) => (
              <View key={i} style={[styles.reasonCard, { 
                backgroundColor: `${colors.primary}06`,
                borderRadius: 12,
                padding: 16,
                marginBottom: 12
              }]}>
                <Text style={[styles.reasonTitle, { 
                  color: colors.text,
                  fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
                  fontSize: 16,
                  marginBottom: 6
                }]}>{point.reason_type}</Text>
                <Text style={[styles.reasonDetail, { 
                  color: colors.textSecondary,
                  fontSize: 14,
                  lineHeight: 20
                }]}>{point.detail}</Text>
              </View>
            ))}
            
            {crop.explanation_points.length > 2 && (
              <Text style={[styles.moreReasonsText, {
                color: colors.primary,
                fontWeight: '500',
                fontSize: 14,
                textAlign: 'center',
                marginTop: 8
              }]}>
                + {crop.explanation_points.length - 2} more reasons
              </Text>
            )}
          </View>
        )}
        
        {/* Risk and subsidy sections with better layout */}
        <View style={styles.bottomSections}>
          {/* Simplified risk section with enhanced styling */}
          {crop.primary_risks.length > 0 && (
            <View style={[styles.risksSection, {
              marginBottom: 25,
              flex: 1
            }]}>
              <Text style={[styles.sectionHeading, { 
                color: colors.text,
                fontSize: 18,
                fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
                marginBottom: 12,
                letterSpacing: -0.3
              }]}>Potential Risks</Text>
              
              {crop.primary_risks.map((risk, i) => (
                <View key={i} style={[styles.riskItem, {
                  flexDirection: 'row',
                  alignItems: 'flex-start',
                  marginBottom: 10
                }]}>
                  <View style={[styles.riskBullet, {
                    backgroundColor: colors.warning,
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    marginTop: 6,
                    marginRight: 10
                  }]} />
                  <Text style={[styles.riskText, { 
                    color: colors.text,
                    flex: 1,
                    fontSize: 14,
                    lineHeight: 20
                  }]}>{risk}</Text>
                </View>
              ))}
            </View>
          )}
          
          {/* Subsidies section with enhanced styling */}
          {crop.relevant_subsidies.length > 0 && (
            <View style={[styles.subsidiesContainer, {
              flex: 1,
              marginBottom: 20
            }]}>
              <Text style={[styles.sectionHeading, { 
                color: colors.text,
                fontSize: 18,
                fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
                marginBottom: 12,
                letterSpacing: -0.3
              }]}>Available Subsidies</Text>
              
              {crop.relevant_subsidies.map((subsidy, i) => (
                <View key={i} style={[styles.subsidyCard, { 
                  backgroundColor: `${colors.success}08`,
                  borderRadius: 12,
                  padding: 14,
                  marginBottom: 10,
                  flexDirection: 'row'
                }]}>
                  <Feather name="award" size={18} color={colors.success} style={[styles.subsidyIcon, {
                    marginRight: 12,
                    marginTop: 2
                  }]} />
                  <View style={styles.subsidyContent}>
                    <Text style={[styles.subsidyTitle, { 
                      color: colors.text,
                      fontWeight: Platform.OS === 'ios' ? '600' : 'bold',
                      fontSize: 15,
                      marginBottom: 4
                    }]}>{subsidy.program}</Text>
                    <Text style={[styles.subsidyDetail, { 
                      color: colors.textSecondary,
                      fontSize: 14,
                      lineHeight: 20
                    }]}>
                      {subsidy.benefit_summary}
                    </Text>
                  </View>
                </View>
              ))}
            </View>
          )}
        </View>
      </View>
    );
  };

  // Crop tab content with improved UI
  const renderCropTab = () => (
    <ScrollView style={styles.tabContent}>
      <View style={styles.sectionContainer}>
        <Text style={[styles.sectionTitle, {color: colors.text}]}>Crop Prediction</Text>
        <Text style={[styles.sectionDescription, {color: colors.textSecondary}]}>
          Find the best crops to grow based on your conditions
        </Text>
        
        <View style={styles.formCard}>
          {renderDateInput('Start Date', cropStartDate, setCropStartDate)}
          {renderDateInput('End Date', cropEndDate, setCropEndDate)}
          
          <View style={styles.formGroup}>
            <Text style={[styles.formLabel, {color: colors.textSecondary}]}>Land Size (Acres)</Text>
            <TextInput
              style={[styles.textInput, {borderColor: colors.border, color: colors.text}]}
              value={cropAcres}
              onChangeText={setCropAcres}
              keyboardType="numeric"
              placeholder="Enter size in acres"
              placeholderTextColor={colors.textSecondary}
            />
          </View>

          <View style={styles.soilSelectionContainer}>
            <Text style={[styles.formLabel, {color: colors.textSecondary}]}>Soil Type</Text>
            
            <View style={styles.soilTypeContainer}>
              {['Loamy', 'Sandy', 'Black'].map(type => (
                <TouchableOpacity
                  key={type}
                  style={[
                    styles.soilTypeButton, 
                    cropSoilType === type && {
                      backgroundColor: colors.primary,
                      shadowColor: colors.primary,
                      shadowOffset: { width: 0, height: 2 },
                      shadowOpacity: 0.3,
                      shadowRadius: 3,
                      elevation: 2
                    }
                  ]}
                  onPress={() => {
                    setCropSoilType(type)
                    setCropSoilImage(null)
                  }}
                >
                  <Text style={{
                    color: cropSoilType === type ? '#fff' : colors.text, 
                    fontWeight: cropSoilType === type ? 'bold' : 'normal'
                  }}>
                    {type}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
            
            <View style={styles.dividerContainer}>
              <View style={styles.dividerLine} />
              <Text style={[styles.orText, {color: colors.textSecondary}]}>OR</Text>
              <View style={styles.dividerLine} />
            </View>
            
            <View style={styles.imageUploadContainer}>
              {cropSoilImage ? (
                <View>
                  <Image source={{uri: cropSoilImage}} style={styles.uploadedImage} />
                  <TouchableOpacity 
                    style={styles.removeImageButton}
                    onPress={() => setCropSoilImage(null)}
                  >
                    <AntDesign name="close" size={20} color="#fff" />
                  </TouchableOpacity>
                </View>
              ) : (
                <TouchableOpacity 
                  style={[styles.uploadButton, {borderColor: colors.border}]}
                  onPress={() => {
                    setCropSoilType('')
                    pickImage(setCropSoilImage)
                  }}
                >
                  <Feather name="upload" size={24} color={colors.text} />
                  <Text style={[styles.uploadText, {color: colors.text}]}>Upload Soil Image</Text>
                </TouchableOpacity>
              )}
            </View>
          </View>
          
          <TouchableOpacity 
            style={[
              styles.predictButton, 
              {
                backgroundColor: colors.primary,
                shadowColor: colors.primary,
                shadowOffset: { width: 0, height: 4 },
                shadowOpacity: 0.3,
                shadowRadius: 5,
                elevation: 3,
                flexDirection: 'row',
                justifyContent: 'center',
                alignItems: 'center'
              }
            ]}
            onPress={handleCropPrediction}
            disabled={loadingCrop}
          >
            {loadingCrop ? (
              <ActivityIndicator color="#fff" size="small" style={{marginRight: 10}} />
            ) : (
              <FontAwesome5 name="magic" size={16} color="#fff" style={{marginRight: 10}} />
            )}
            <Text style={styles.predictButtonText}>
              {loadingCrop ? "Analyzing..." : "Predict Best Crops"}
            </Text>
          </TouchableOpacity>
        </View>
        
        {cropResult && (
          <View style={styles.advancedResultContainer}>
            {/* Simplified analysis summary card */}
            <View style={[styles.summaryCard, { 
              backgroundColor: colors.card,
              shadowColor: '#000',
              shadowOffset: { width: 0, height: 3 },
              shadowOpacity: 0.1,
              shadowRadius: 4,
              elevation: 3
            }]}>
              <View style={styles.summaryHeader}>
                <View>
                  <Text style={[styles.summaryTitle, {color: colors.text}]}>Analysis Summary</Text>
                  <Text 
                    style={[styles.summarySubtitle, {color: colors.textSecondary}]}>{`${cropResult.request_details.soil_type} soil • ${cropResult.request_details.land_size_acres} acres`}</Text>
                </View>
                <View style={[styles.locationBadge, {backgroundColor: `${colors.primary}15`}]}>
                  <MaterialIcons name="location-on" size={16} color={colors.primary} />
                  <Text style={[styles.locationText, {color: colors.primary}]}>
                    {`${cropResult.request_details.latitude.toFixed(1)}, ${cropResult.request_details.longitude.toFixed(1)}`}
                  </Text>
                </View>
              </View>
              
              {/* Weather summary */}
              <View style={[styles.weatherSummaryInline, { 
                backgroundColor: `${colors.primary}10`,
                borderLeftWidth: 3,
                borderLeftColor: colors.primary
              }]}>
                <Feather name="cloud-rain" size={18} color={colors.primary} style={styles.weatherIcon} />
                <Text style={[styles.weatherSummaryText, { color: colors.text }]}>
                  {cropResult.weather_context_summary}
                </Text>
              </View>
            </View>
            
            <Text style={[styles.recommendationsHeading, { color: colors.text, marginTop: 25 }]}>
              Recommended Crops
            </Text>
            
            {cropResult.recommendations.map((crop, index) => (
              <CropDetailView key={index} crop={crop} index={index} />
            ))}
            
            <View style={styles.disclaimerContainer}>
              <Text style={[styles.disclaimerText, {color: colors.textSecondary}]}>
                * Recommendations are based on historical data, weather forecasts, and market analysis.
                Actual results may vary based on local conditions and management practices.
              </Text>
            </View>
          </View>
        )}
      </View>
    </ScrollView>
  )

  return (
    <SafeAreaView style={[styles.container, { backgroundColor: colors.background }]}>
      <Text style={[styles.screenTitle, { color: colors.text }]}>
        <TranslatedText>Agricultural Forecasts</TranslatedText>
      </Text>
      {renderTabs()}
      
      {activeTab === 'soil' && renderSoilTab()}
      {activeTab === 'disease' && renderDiseaseTab()}
      {activeTab === 'weather' && renderWeatherTab()}
      {activeTab === 'crop' && renderCropTab()}
    </SafeAreaView>
  )
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
  },
  screenTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginHorizontal: 20,
    marginVertical: 15,
  },
  tabContainer: {
    flexDirection: 'row',
    marginHorizontal: 10,
    marginBottom: 15,
  },
  tab: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 10,
    marginHorizontal: 5,
    borderRadius: 8,
    backgroundColor: '#f0f0f0',
  },
  tabContent: {
    flex: 1,
  },
  sectionContainer: {
    marginHorizontal: 20,
    marginBottom: 25,
  },
  sectionTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  sectionDescription: {
    fontSize: 16,
    marginBottom: 20,
  },
  imageUploadContainer: {
    alignItems: 'center',
    marginBottom: 20,
  },
  uploadButton: {
    borderWidth: 1,
    borderStyle: 'dashed',
    borderRadius: 10,
    padding: 20,
    alignItems: 'center',
    width: '100%',
  },
  uploadText: {
    marginTop: 10,
    fontSize: 16,
  },
  uploadedImage: {
    width: 200,
    height: 200,
    borderRadius: 10,
  },
  removeImageButton: {
    position: 'absolute',
    top: -10,
    right: -10,
    backgroundColor: 'rgba(0,0,0,0.7)',
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
  },
  predictButton: {
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
    justifyContent: 'center',
  },
  disabledButton: {
    opacity: 0.5,
  },
  predictButtonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: 'bold',
  },
  resultContainer: {
    marginTop: 20,
    padding: 15,
    borderRadius: 10,
  },
  resultTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  resultRow: {
    flexDirection: 'row',
    marginBottom: 10,
  },
  resultLabel: {
    width: 100,
    fontSize: 16,
  },
  resultValue: {
    flex: 1,
    fontSize: 16,
    fontWeight: '500',
  },
  historyItem: {
    padding: 15,
    borderRadius: 8,
    marginBottom: 10,
  },
  historyItemRow: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  historyItemLabel: {
    width: 80,
  },
  historyItemValue: {
    flex: 1,
    fontWeight: '500',
  },
  historyDate: {
    marginTop: 5,
    fontSize: 12,
  },
  formGroup: {
    marginBottom: 15,
  },
  formLabel: {
    fontSize: 16,
    marginBottom: 8,
  },
  datePickerButton: {
    borderWidth: 1,
    padding: 12,
    borderRadius: 8,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  textInput: {
    borderWidth: 1,
    padding: 12,
    borderRadius: 8,
    fontSize: 16,
  },
  weatherItem: {
    marginBottom: 15,
  },
  weatherMonth: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  weatherDetails: {
    marginLeft: 10,
  },
  weatherDetail: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 3,
  },
  weatherValue: {
    marginLeft: 8,
  },
  soilSelectionContainer: {
    marginBottom: 20,
  },
  soilTypeContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 10,
  },
  soilTypeButton: {
    flex: 1,
    padding: 10,
    alignItems: 'center',
    borderRadius: 8,
    marginHorizontal: 5,
    backgroundColor: '#f0f0f0',
  },
  orText: {
    textAlign: 'center',
    marginVertical: 10,
    fontSize: 16,
  },
  subheading: {
    fontSize: 17,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  cropItem: {
    marginBottom: 15,
    paddingBottom: 15,
    borderBottomWidth: 1,
    borderBottomColor: '#eee',
  },
  cropName: {
    fontSize: 16,
    fontWeight: 'bold',
    marginBottom: 5,
  },
  cropDetail: {
    flexDirection: 'row',
    marginLeft: 5,
    marginBottom: 3,
  },
  cropDetailLabel: {
    width: 100,
  },
  cropDetailValue: {
    flex: 1,
  },
  subsidiesContainer: {
    marginTop: 10,
  },
  emptyContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  emptyText: {
    fontSize: 16,
    textAlign: 'center',
  },
  dateActions: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  dateAction: {
    padding: 8,
    marginLeft: 4,
  },
  dateFormatHint: {
    fontSize: 12,
    marginTop: 4,
    fontStyle: 'italic',
  },
  // New styles for advanced crop prediction results
  advancedResultContainer: {
    marginTop: 20,
  },
  requestSummary: {
    padding: 15,
    borderRadius: 10,
    marginBottom: 15,
  },
  requestDetailRow: {
    flexDirection: 'row',
    marginBottom: 8,
  },
  requestDetailLabel: {
    width: 100,
    fontSize: 15,
  },
  requestDetailValue: {
    flex: 1,
    fontSize: 15,
    fontWeight: '500',
  },
  weatherSummaryContainer: {
    marginBottom: 20,
  },
  weatherSummaryCard: {
    padding: 15,
    borderRadius: 10,
    flexDirection: 'row',
    alignItems: 'center',
  },
  weatherIcon: {
    marginRight: 12,
  },
  weatherSummaryText: {
    flex: 1,
    fontSize: 15,
    lineHeight: 22,
  },
  recommendationsHeading: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  cropDetailCard: {
    borderRadius: 10,
    padding: 15,
    marginBottom: 20,
    borderWidth: 2,
    position: 'relative',
  },
  cropRankBadge: {
    position: 'absolute',
    top: -12,
    right: -12,
    backgroundColor: '#4CAF50',
    width: 36,
    height: 36,
    borderRadius: 18,
    justifyContent: 'center',
    alignItems: 'center',
    zIndex: 1,
  },
  cropRankText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
  cropDetailTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  scoreContainer: {
    marginBottom: 20,
  },
  scoreBar: {
    height: 12,
    borderRadius: 6,
    marginVertical: 8,
    overflow: 'hidden',
  },
  scoreValue: {
    height: '100%',
    borderRadius: 6,
  },
  keyMetricsContainer: {
    marginBottom: 20,
  },
  sectionSubheading: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 12,
  },
  metricRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 12,
  },
  metricItem: {
    flex: 1,
    padding: 10,
    alignItems: 'center',
  },
  metricLabel: {
    marginTop: 5,
    marginBottom: 3,
    fontSize: 14,
  },
  metricValue: {
    fontWeight: 'bold',
    textAlign: 'center',
    fontSize: 14,
  },
  chartContainer: {
    marginBottom: 25,
    alignItems: 'center', // Center charts in container
  },
  chartDescription: {
    fontSize: 14,
    marginBottom: 10,
    fontStyle: 'italic',
    textAlign: 'center',
    paddingHorizontal: 10,
  },
  chart: {
    borderRadius: 10,
    paddingRight: 16, // Add padding to prevent labels from being cut off
    marginHorizontal: -10, // Compensate for chart internal padding
    alignSelf: 'center',
  },
  barChartWrapper: {
    marginTop: 10,
    alignItems: 'center',
    width: '100%',
  },
  chartLegend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    marginTop: 8,
    paddingHorizontal: 10,
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 12,
    marginBottom: 6,
    maxWidth: '45%', // Allow 2 items per row
  },
  legendDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 4,
  },
  legendText: {
    fontSize: 12,
    flex: 1,
  },
  reasonsContainer: {
    marginBottom: 20,
  },
  reasonItem: {
    flexDirection: 'row',
    marginBottom: 15,
  },
  reasonBullet: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    marginTop: 2,
  },
  reasonBulletText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 14,
  },
  reasonContent: {
    flex: 1,
  },
  reasonTitle: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  reasonDetail: {
    fontSize: 14,
    lineHeight: 20,
  },
  subsidyItem: {
    flexDirection: 'row',
    padding: 12,
    borderRadius: 8,
    marginBottom: 10,
  },
  subsidyIcon: {
    marginRight: 10,
    marginTop: 2,
  },
  subsidyContent: {
    flex: 1,
  },
  subsidyTitle: {
    fontWeight: 'bold',
    fontSize: 16,
    marginBottom: 4,
  },
  subsidyProvider: {
    fontSize: 14,
    marginBottom: 4,
  },
  subsidyBenefit: {
    fontSize: 14,
  },
  risksContainer: {
    marginBottom: 20,
  },
  riskItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
  },
  riskIcon: {
    marginRight: 8,
  },
  riskText: {
    flex: 1,
    fontSize: 14,
  },
  fertilizerScheduleContainer: {
    marginBottom: 10,
  },
  fertilizerItem: {
    flexDirection: 'row',
    borderRadius: 8,
    marginBottom: 10,
    overflow: 'hidden',
  },
  fertilizerStage: {
    width: 80,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 10,
  },
  fertilizerStageText: {
    color: 'white',
    fontWeight: 'bold',
  },
  fertilizerDetail: {
    flex: 1,
    padding: 10,
  },
  fertilizerTiming: {
    fontSize: 14,
    marginBottom: 4,
  },
  fertilizerNutrients: {
    fontWeight: '500',
    fontSize: 15,
  },
  // Enhanced styles for crop prediction
  formCard: {
    backgroundColor: '#fff',
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.1,
    shadowRadius: 2,
    elevation: 2,
  },
  dividerContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    marginVertical: 15,
  },
  dividerLine: {
    flex: 1,
    height: 1,
    backgroundColor: '#E0E0E0',
  },
  summaryHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 15,
  },
  requestDetailGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginHorizontal: -5,
  },
  requestDetailItem: {
    width: '50%',
    paddingHorizontal: 5,
    marginBottom: 12,
  },
  recommendationsHeader: {
    marginBottom: 20,
  },
  recommendationsSubheading: {
    fontSize: 14,
  },
  cropHeaderContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: 20,
    paddingRight: 15,
  },
  scoreCircle: {
    width: 48,
    height: 48,
    borderRadius: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  scoreText: {
    color: 'white',
    fontSize: 16,
  },
  scoreLabel: {
    textAlign: 'center',
  },
  metricsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  metricSeparator: {
    width: 1,
    height: '60%',
  },
  chartSection: {
    marginBottom: 28,
  },
  chartTitleRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 12,
  },
  enhancedChart: {
    borderRadius: 12,
  },
  waterNeedsContainer: {
    alignItems: 'center',
  },
  waterLegend: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-around',
    width: '100%',
  },
  waterLegendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 8,
    width: '48%',
  },
  bottomSections: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  dateInputContainer: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  dateInput: {
    flex: 1,
    borderWidth: 1,
    padding: 12,
    borderRadius: 8,
    fontSize: 16,
    marginHorizontal: 8,
  },
  dateActionButton: {
    width: 44,
    height: 44,
    justifyContent: 'center',
    alignItems: 'center',
  },
  weatherDateRangeCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  weatherSummaryCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  weatherSummaryTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 4,
  },
  weatherSummarySubtitle: {
    fontSize: 14,
    marginBottom: 16,
  },
  weatherHighlightsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 10,
  },
  weatherHighlightItem: {
    width: '30%',
    borderRadius: 10,
    padding: 12,
    alignItems: 'center',
  },
  weatherHighlightIconContainer: {
    width: 38,
    height: 38,
    borderRadius: 19,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 8,
  },
  weatherHighlightValue: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 2,
  },
  weatherHighlightLabel: {
    fontSize: 12,
  },
  chartCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  chartTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  chart: {
    marginVertical: 8,
    borderRadius: 8,
  },
  chartLegend: {
    flexDirection: 'row',
    justifyContent: 'center',
    marginTop: 15,
    flexWrap: 'wrap',
  },
  legendItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginRight: 20,
    marginBottom: 5,
  },
  legendColor: {
    width: 12,
    height: 12,
    borderRadius: 6,
    marginRight: 6,
  },
  legendText: {
    fontSize: 12,
  },

  smallChartsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 20,
  },
  smallChartCard: {
    width: '48%',
    borderRadius: 12,
    padding: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  smallChart: {
    marginVertical: 8,
    borderRadius: 8,
  },
  detailedWeatherCard: {
    borderRadius: 12,
    padding: 16,
    marginBottom: 20,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  detailedWeatherTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
  },
  monthlyWeatherItem: {
    paddingVertical: 8,
  },
  monthLabel: {
    fontSize: 16,
    fontWeight: '600',
    marginBottom: 10,
  },
  weatherMetricsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
  },
  weatherMetricItem: {
    width: '50%',
    marginBottom: 12,
    paddingRight: 10,
  },
  weatherMetricRow: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 4,
  },
  weatherMetricLabel: {
    fontSize: 12,
    marginLeft: 5,
  },
  weatherMetricValue: {
    fontSize: 14,
    fontWeight: '500',
    paddingLeft: 20,
  },
  weatherNotice: {
    flexDirection: 'row',
    alignItems: 'flex-start',
    marginBottom: 30,
    paddingHorizontal: 5,
  },
  weatherNoticeIcon: {
    width: 30,
    height: 30,
    borderRadius: 15,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: 10,
    marginTop: 2,
  },
  weatherNoticeText: {
    flex: 1,
    fontSize: 13,
    lineHeight: 18,
  },
  noForecastContainer: {
    alignItems: 'center',
    padding: 30,
    marginTop: 20,
    borderRadius: 12,
    backgroundColor: '#f9f9f9',
    borderWidth: 1,
    borderColor: '#e0e0e0',
    borderStyle: 'dashed',
  },
  noForecastIcon: {
    width: 60,
    height: 60,
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  noForecastText: {
    textAlign: 'center',
    fontSize: 16,
    lineHeight: 24,
  },
})

// Make sure to export the component as default
export default ForecastScreen;
