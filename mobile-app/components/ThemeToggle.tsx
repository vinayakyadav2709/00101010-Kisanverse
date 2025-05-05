import React from 'react';
import { TouchableOpacity, StyleSheet, View, Animated } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { Moon, Sun } from 'react-native-feather';

export const ThemeToggle: React.FC = () => {
  const { theme, toggleTheme, colors } = useTheme();
  const animatedValue = React.useRef(new Animated.Value(theme === 'dark' ? 1 : 0)).current;

  React.useEffect(() => {
    Animated.timing(animatedValue, {
      toValue: theme === 'dark' ? 1 : 0,
      duration: 200,
      useNativeDriver: false,
    }).start();
  }, [theme]);

  const togglerTranslate = animatedValue.interpolate({
    inputRange: [0, 1],
    outputRange: [2, 22],
  });

  return (
    <TouchableOpacity
      onPress={toggleTheme}
      style={[
        styles.container,
        { backgroundColor: theme === 'dark' ? colors.backgroundSecondary : colors.backgroundSecondary }
      ]}
      activeOpacity={0.8}
    >
      <Animated.View
        style={[
          styles.toggler,
          { 
            backgroundColor: theme === 'dark' ? colors.primary : colors.primary,
            transform: [{ translateX: togglerTranslate }],
          },
        ]}
      >
        {theme === 'dark' ? (
          <Moon width={16} height={16} stroke="#fff" />
        ) : (
          <Sun width={16} height={16} stroke="#fff" />
        )}
      </Animated.View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    width: 50,
    height: 28,
    borderRadius: 14,
    padding: 2,
    justifyContent: 'center',
  },
  toggler: {
    width: 24,
    height: 24,
    borderRadius: 12,
    justifyContent: 'center',
    alignItems: 'center',
  },
});
