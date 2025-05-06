import React, { ReactNode } from 'react';
import { Text, StyleSheet, StyleProp, TextStyle } from 'react-native';
import { useTheme } from '../context/ThemeContext';

interface TypographyProps {
  children: ReactNode;
  variant?: 'heading' | 'subheading' | 'bodyLarge' | 'body' | 'caption' | 'small' | 'headingLarge';
  color?: 'primary' | 'secondary' | 'text' | 'textSecondary' | 'accent' | 'error' | 'warning' | 'success' | 'info';
  style?: StyleProp<TextStyle>;
  centered?: boolean;
}

export const Typography: React.FC<TypographyProps> = ({
  children,
  variant = 'body',
  color = 'text',
  style,
  centered = false,
}) => {
  const { colors, typography } = useTheme();
  
  return (
    <Text
      style={[
        typography[variant],
        { color: colors[color] },
        centered && { textAlign: 'center' },
        style,
      ]}
    >
      {children}
    </Text>
  );
};
