import { create } from 'zustand';

// Define the structure of our store
interface TranslationStore {
  language: string;
  translations: Record<string, Record<string, string>>;
  isLoading: boolean;
  setLanguage: (language: string) => void;
  setTranslations: (language: string, translations: Record<string, string>) => void;
  setIsLoading: (isLoading: boolean) => void;
  translate: (key: string, fallback?: string) => string;
}

// Language translation mapping
export const languages = [
  { code: 'en', name: 'English' },
  { code: 'hi', name: 'हिंदी' },  // Hindi
  { code: 'mr', name: 'मराठी' },  // Marathi
];

// Initial translations for common UI elements
const initialTranslations = {
  en: {
    // Dashboard
    "Hello": "Hello",
    "Welcome to your farm dashboard": "Welcome to your farm dashboard",
    "Dashboard": "Dashboard",
    "Forecast": "Forecast",
    "Marketplace": "Marketplace",
    "Schemes": "Schemes",
    "Planning": "Planning",
    "My Fields": "My Fields",
    "See All": "See All",
    "Add Field": "Add Field",
    "Commodities and Food": "Commodities and Food",
    "Recommended Crops": "Recommended Crops",
    "View All": "View All",
    "Action Items": "Action Items",
    
    // Marketplace
    "New Listing": "New Listing",
    "Listings": "Listings",
    "Contracts": "Contracts",
    "Subsidies": "Subsidies",
    "Contact Seller": "Contact Seller",
    "Place Bid": "Place Bid",
    
    // Weather & Forecast
    "Weather Forecast": "Weather Forecast",
    "Crop Recommendations": "Crop Recommendations",
    "Disease Detection": "Disease Detection",
    "Soil Analysis": "Soil Analysis",
    "Temperature": "Temperature",
    "Humidity": "Humidity",
    "Wind": "Wind",
    "Rain": "Rain",
    "Sunrise": "Sunrise",
    "Sunset": "Sunset",
    
    // Statuses
    "Pending": "Pending",
    "Accepted": "Accepted",
    "Rejected": "Rejected",
    "Completed": "Completed",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
    
    // Actions
    "Complete": "Complete",
    "Search here...": "Search here...",
    "Get Started": "Get Started",
    "Save": "Save",
    "Cancel": "Cancel",
    "Submit": "Submit",
    "Apply": "Apply",
    "Back": "Back",
  },
  hi: {},  // Will be populated dynamically
  mr: {}   // Will be populated dynamically
};

// Create the Zustand store
export const useTranslationStore = create<TranslationStore>((set, get) => ({
  language: 'en',
  translations: initialTranslations,
  isLoading: false,
  
  setLanguage: (language) => set({ language }),
  
  setTranslations: (language, translations) => 
    set((state) => ({
      translations: {
        ...state.translations,
        [language]: translations
      }
    })),
  
  setIsLoading: (isLoading) => set({ isLoading }),
  
  translate: (key, fallback) => {
    const { language, translations } = get();
    
    // Try to get the translation for the current language
    if (translations[language] && translations[language][key]) {
      return translations[language][key];
    }
    
    // Fallback to English if translation not found
    if (language !== 'en' && translations.en && translations.en[key]) {
      return translations.en[key];
    }
    
    // Return the fallback or the key itself if no translation found
    return fallback || key;
  }
}));

// Function to translate a batch of text
export const translateBatch = async (
  texts: string[], 
  targetLang: string
): Promise<Record<string, string>> => {
  if (targetLang === 'en') {
    // For English, just return identity mapping
    return texts.reduce((acc, text) => {
      acc[text] = text;
      return acc;
    }, {} as Record<string, string>);
  }
  
  try {
    const response = await fetch('https://cfcc-183-87-252-69.ngrok-free.app/translate', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        text: texts.join('\n'),
        targetLang
      }),
    });
    
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Translation error:', error);
    return {};
  }
};
