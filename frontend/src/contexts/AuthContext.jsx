import React, { createContext, useState, useContext, useEffect } from 'react';
import { api } from '../services/api';

const AuthContext = createContext(null);

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkAuth();
  }, []);

  const checkAuth = async () => {
    try {
      const token = localStorage.getItem('token');
      console.log('Auth check - stored token:', token ? token.substring(0, 20) + '...' : 'no token');
      
      if (token) {
        // Update API headers with formatted token
        const formattedToken = `Bearer ${token}`;
        api.defaults.headers.common['Authorization'] = formattedToken;
        console.log('Auth check - Authorization header:', formattedToken.substring(0, 30) + '...');
        
        console.log('Fetching user data for auth check...');
        const userData = await fetchUserData();
        console.log('Auth check successful:', userData);
        setUser(userData);
      } else {
        console.log('No token found during auth check');
        setUser(null);
      }
    } catch (error) {
      console.error('Auth check failed:', {
        error: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers
      });
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      setUser(null);
    } finally {
      setLoading(false);
    }
  };

  const login = async (email, password) => {
    try {
      console.log('Login attempt:', { email });
      
      // Clear any existing token
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      
      // Use URLSearchParams for proper form encoding
      const formData = new URLSearchParams();
      formData.append('username', email);  // OAuth2 expects 'username'
      formData.append('password', password);
      
      console.log('Sending login request...');
      const response = await api.post('/auth/login', formData.toString(), {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded'
        }
      });
      
      const { access_token, token_type } = response.data;
      console.log('Login response:', { 
        token_type,
        access_token: access_token ? access_token.substring(0, 20) + '...' : 'missing'
      });
      
      if (!access_token) {
        throw new Error('No access token received from server');
      }
      
      // Store raw token without Bearer prefix
      localStorage.setItem('token', access_token);
      
      // Update API headers with formatted token
      const formattedToken = `Bearer ${access_token}`;
      api.defaults.headers.common['Authorization'] = formattedToken;
      
      // Verify token is stored and formatted correctly
      const storedToken = localStorage.getItem('token');
      console.log('Stored token:', storedToken ? storedToken.substring(0, 20) + '...' : 'missing');
      console.log('Authorization header:', api.defaults.headers.common['Authorization'].substring(0, 30) + '...');
      
      // Fetch user data after successful login
      console.log('Fetching user data...');
      const userData = await fetchUserData();
      console.log('User data fetched:', userData);
      setUser(userData);
      
      return { success: true };
    } catch (error) {
      console.error('Login failed:', {
        error: error.message,
        response: error.response?.data,
        status: error.response?.status,
        headers: error.response?.headers
      });
      throw new Error(error.response?.data?.detail || 'Login failed. Please try again.');
    }
  };

  const register = async (email, password, username, full_name) => {
    try {
      const response = await api.post('/auth/signup', {
        email,
        password,
        username,
        full_name
      });
      
      // After registration, log the user in
      return await login(email, password);
    } catch (error) {
      console.error('Registration failed:', error);
      throw new Error(error.response?.data?.detail || 'Registration failed. Please try again.');
    }
  };

  const logout = () => {
    localStorage.removeItem('token');
    delete api.defaults.headers.common['Authorization'];
    setUser(null);
  };

  const value = {
    user,
    login,
    logout,
    register,
    loading,
  };

  return (
    <AuthContext.Provider value={value}>
      {!loading && children}
    </AuthContext.Provider>
  );
};

// Helper function to fetch user data
const fetchUserData = async () => {
  try {
    const response = await api.get('/users/me');
    return response.data;
  } catch (error) {
    console.error('Failed to fetch user data:', error);
    throw error;
  }
};
