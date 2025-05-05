import { create } from "zustand"
import { persist, createJSONStorage } from "zustand/middleware"
import AsyncStorage from "@react-native-async-storage/async-storage"

export interface User {
  $id?: string
  name: string
  email: string
  role: string  // Now will support "farmer" or "buyer"
  address: string
  zipcode: string  // Add zipcode field
  $createdAt?: string
  $updatedAt?: string
}

interface UserState {
  user: User | null
  isLoggedIn: boolean
  isLoading: boolean
  error: string | null
  
  // Actions
  registerUser: (userData: Omit<User, "$id" | "$createdAt" | "$updatedAt">) => Promise<void>
  loginUser: (email: string, password: string) => Promise<void>
  logoutUser: () => void
  fetchUserByEmail: (email: string) => Promise<User | null>
  clearError: () => void
}

export const useUserStore = create<UserState>()(
  persist(
    (set, get) => ({
      user: null,
      isLoggedIn: false,
      isLoading: false,
      error: null,

      registerUser: async (userData) => {
        try {
          console.log('Starting user registration process', userData);
          set({ isLoading: true, error: null });
          
          // Use the role provided or default to "farmer"
          const userDataWithRole = {
            ...userData,
            role: userData.role || "farmer",
            zipcode: userData.zipcode || "" // Ensure zipcode is included
          };
          
          console.log('Sending registration request with data:', userDataWithRole);
          const response = await fetch('https://4f70-124-66-175-40.ngrok-free.app/users', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(userDataWithRole),
          });
          
          const data = await response.json();
          console.log('Registration API response:', data);
          
          if (!response.ok) {
            throw new Error(data.message || 'Registration failed');
          }
          
          set({ 
            user: data, 
            isLoggedIn: true, 
            isLoading: false 
          });
          console.log('User registered successfully:', data);
        } catch (error) {
          console.error('Registration error:', error);
          set({ 
            error: error instanceof Error ? error.message : 'An unknown error occurred', 
            isLoading: false 
          });
        }
      },
      
      loginUser: async (email, password) => {
        try {
          console.log('Starting login process for:', email);
          set({ isLoading: true, error: null });
          
          // In a real app, you would validate credentials against a backend
          // For now, we're just fetching the user by email
          console.log('Fetching user data for:', email);
          const response = await fetch(`https://4f70-124-66-175-40.ngrok-free.app/users/${email}`);
          const userData = await response.json();
          console.log('Login API response:', userData);
          
          if (!response.ok) {
            throw new Error(userData.message || 'Login failed');
          }
          
          // Since we don't have real password verification, we'll accept any password for now
          // In a production app, you would verify the password on the server
          
          // Check user's role and log it
          const userRole = userData.role || 'unknown';
          console.log('User role detected:', userRole);
          
          set({ 
            user: userData, 
            isLoggedIn: true, 
            isLoading: false 
          });
          console.log('User logged in successfully:', userData);
          
          // Return the role so it can be used for routing in the login component
          return userRole;
        } catch (error) {
          console.error('Login error:', error);
          set({ 
            error: error instanceof Error ? error.message : 'An unknown error occurred', 
            isLoading: false 
          });
        }
      },
      
      logoutUser: () => {
        console.log('Logging out user');
        set({ user: null, isLoggedIn: false });
      },
      
      fetchUserByEmail: async (email) => {
        try {
          console.log('Fetching user data for:', email);
          const response = await fetch(`https://4f70-124-66-175-40.ngrok-free.app/${email}`);
          
          if (!response.ok) {
            console.log('User not found:', email);
            return null;
          }
          
          const userData = await response.json();
          console.log('User data retrieved:', userData);
          return userData;
        } catch (error) {
          console.error('Error fetching user:', error);
          return null;
        }
      },
      
      clearError: () => set({ error: null }),
    }),
    {
      name: 'user-storage',
      storage: createJSONStorage(() => AsyncStorage),
    }
  )
);
