import React, { ReactNode } from 'react';
import { View, StyleSheet, StyleProp, ViewStyle } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface CardProps {
  children: ReactNode;
  style?: StyleProp<ViewStyle>;
  variant?: 'default' | 'elevated';
  noPadding?: boolean;
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  style, 
  variant = 'default',
  noPadding = false 
}) => {
  const { colors, radius, spacing } = useTheme();
  
  return (
    <View 
      style={[
        styles.card,
        !noPadding && { padding: spacing.md },
        { 
          backgroundColor: colors.card,
          borderColor: colors.border,
          borderRadius: radius.lg,
          ...variant === 'elevated' && {
            shadowColor: colors.shadow,
            shadowOffset: { width: 0, height: 2 },
            shadowOpacity: 0.1,
            shadowRadius: 8,
            elevation: 2,
          }
        },
        style
      ]}
    >
      {children}
    </View>
  );
};

const styles = StyleSheet.create({
  card: {
    borderWidth: 1,
    overflow: 'hidden',
  },
});
