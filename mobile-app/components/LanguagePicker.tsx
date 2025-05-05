import React, { useState } from 'react';
import { View, TouchableOpacity, Text, Modal, StyleSheet } from 'react-native';
import { useTheme } from '../context/ThemeContext';
import { Globe } from 'react-native-feather';
import { useLanguageStore } from '../store/languageStore';

export const LanguagePicker = () => {
  const { colors, radius, spacing } = useTheme();
  const [modalVisible, setModalVisible] = useState(false);
  
  // Use primitive selectors to avoid unnecessary re-renders
  const language = useLanguageStore(state => state.language);
  const setLanguage = useLanguageStore(state => state.setLanguage);

  const languages = [
    { code: 'en', name: 'English' },
    { code: 'hi', name: 'हिंदी' },
    { code: 'mr', name: 'मराठी' }
  ];

  const handleLanguageChange = (langCode: string) => {
    // Set the language in the store
    setLanguage(langCode);
    // Close the modal
    setModalVisible(false);
  };

  return (
    <>
      <TouchableOpacity onPress={() => setModalVisible(true)}>
        <View style={styles.languageButton}>
          <Globe width={20} height={20} stroke={colors.primary} />
          <Text style={[styles.currentLanguage, { color: colors.primary }]}>
            {language.toUpperCase()}
          </Text>
        </View>
      </TouchableOpacity>

      <Modal
        animationType="fade"
        transparent={true}
        visible={modalVisible}
        onRequestClose={() => setModalVisible(false)}
      >
        <View style={styles.centeredView}>
          <View style={[styles.modalView, { 
            backgroundColor: colors.card,
            borderRadius: radius.lg
          }]}>
            <Text style={[styles.modalTitle, { color: colors.text }]}>Select Language</Text>
            
            {languages.map((lang) => (
              <TouchableOpacity
                key={lang.code}
                style={[
                  styles.languageOption,
                  language === lang.code && { 
                    backgroundColor: colors.primary + '20',
                    borderColor: colors.primary,
                  }
                ]}
                onPress={() => handleLanguageChange(lang.code)}
              >
                <Text style={[
                  styles.languageText, 
                  { color: language === lang.code ? colors.primary : colors.text }
                ]}>
                  {lang.name}
                </Text>
              </TouchableOpacity>
            ))}
            
            <TouchableOpacity
              style={[styles.closeButton, { backgroundColor: colors.backgroundSecondary }]}
              onPress={() => setModalVisible(false)}
            >
              <Text style={{ color: colors.text }}>Close</Text>
            </TouchableOpacity>
          </View>
        </View>
      </Modal>
    </>
  );
};

const styles = StyleSheet.create({
  centeredView: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(0, 0, 0, 0.5)',
  },
  modalView: {
    margin: 20,
    padding: 20,
    elevation: 5,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    width: '80%',
  },
  modalTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 15,
    textAlign: 'center',
  },
  languageOption: {
    padding: 15,
    marginVertical: 5,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: 'transparent',
  },
  languageText: {
    fontSize: 16,
    textAlign: 'center',
  },
  closeButton: {
    marginTop: 20,
    padding: 15,
    borderRadius: 8,
    alignItems: 'center',
  },
  languageButton: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  currentLanguage: {
    marginLeft: 4,
    fontSize: 12,
    fontWeight: 'bold',
  }
});
