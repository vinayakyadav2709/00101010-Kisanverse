import React from 'react';
import { Text, TextProps } from 'react-native';
import { useLanguageStore } from '../store/languageStore';

interface TranslatedTextProps extends TextProps {
  children?: React.ReactNode;
  translationKey: string;
}

export const TranslatedText: React.FC<TranslatedTextProps> = ({
  children,
  translationKey,
  ...props
}) => {
  const translate = useLanguageStore(state => state.translate);
  const translatedText = translate(translationKey);
  
  // Helper function to safely convert any value to string
  const safeToString = (value: any): string => {
    if (value === null || value === undefined) {
      return '';
    } else if (typeof value === 'string') {
      return value;
    } else if (typeof value === 'object') {
      // Check if it's a React element
      if (React.isValidElement(value)) {
        // For React elements, return empty string and let it render as a child
        return '';
      }
      try {
        // Try to stringify objects
        return JSON.stringify(value);
      } catch (e) {
        return '[Object]';
      }
    }
    // Convert numbers, booleans, etc. to strings
    return String(value);
  };

  // Render the translated text if available, otherwise use children
  return (
    <Text {...props}>
      {translatedText ? safeToString(translatedText) : children}
    </Text>
  );
};
