import axios from 'axios';

// Create axios instance with base configuration
export const api = axios.create({
  baseURL: 'http://localhost:8001/api/v1',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add request interceptor to attach token
api.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token');
    console.log('Interceptor - raw token:', token ? token.substring(0, 20) + '...' : 'no token');
    
    if (token) {
      // Ensure token is properly formatted with Bearer prefix
      const formattedToken = `Bearer ${token}`;
      config.headers['Authorization'] = formattedToken;
      
      console.log('Interceptor - headers:', {
        Authorization: formattedToken.substring(0, 30) + '...',
        'Content-Type': config.headers['Content-Type'],
        url: config.url
      });
    } else {
      console.log('No token found in localStorage');
      // Remove Authorization header if no token
      delete config.headers['Authorization'];
    }
    return config;
  },
  (error) => {
    console.error('Request interceptor error:', error.message);
    return Promise.reject(error);
  }
);

// Add response interceptor to handle errors
api.interceptors.response.use(
  (response) => {
    console.log('Response:', {
      url: response.config.url,
      status: response.status,
      headers: response.headers,
      data: response.data
    });
    return response;
  },
  (error) => {
    console.error('API Error:', {
      url: error.config?.url,
      status: error.response?.status,
      data: error.response?.data,
      message: error.message,
      headers: {
        request: error.config?.headers,
        response: error.response?.headers
      }
    });
    
    if (error.response?.status === 401) {
      console.log('Unauthorized, clearing token');
      localStorage.removeItem('token');
      delete api.defaults.headers.common['Authorization'];
      // Only redirect if not already on login page
      if (!window.location.pathname.includes('/login')) {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);
