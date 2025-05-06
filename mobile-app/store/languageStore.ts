import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"
import AsyncStorage from "@react-native-async-storage/async-storage"

interface LanguageState {
  language: string
  translations: Record<string, Record<string, string>>
  setLanguage: (language: string) => void
  addTranslations: (translations: Record<string, Record<string, string>>) => void
  translate: (key: string) => string | undefined
}

export const useLanguageStore = create<LanguageState>()(
  persist(
    (set, get) => ({
      language: 'en', // Default language
      translations: {},

      setLanguage: (language) => {
        set({ language })
      },

      addTranslations: (newTranslations) => {
        set((state) => ({
          translations: {
            ...state.translations,
            ...newTranslations
          }
        }))
      },

      translate: (key) => {
        const { language, translations } = get()
        
        // First try to get the translation in the current language
        if (translations[language]?.[key]) {
          return translations[language][key]
        }
        
        // Fallback to English
        if (language !== 'en' && translations['en']?.[key]) {
          return translations['en'][key]
        }
        
        // If no translation is found, return undefined
        return undefined
      }
    }),
    {
      name: 'language-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
)
