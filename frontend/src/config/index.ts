// API Configuration
export const API_CONFIG = {
  baseUrl: 'http://localhost:8001/api/v1',
  timeout: 30000, // 30 seconds
  retryAttempts: 3,
} as const;

// Authentication Configuration
export const AUTH_CONFIG = {
  tokenKey: 'thunderai_token',
  refreshTokenKey: 'thunderai_refresh_token',
  tokenExpiry: 8 * 24 * 60 * 60 * 1000, // 8 days in milliseconds
} as const;

// Application Configuration
export const APP_CONFIG = {
  appName: 'ThunderAI',
  version: '1.0.0',
  defaultTheme: 'light',
  defaultLanguage: 'en',
} as const;

// Feature Flags Configuration
export const FEATURES = {
  enableApiKeys: true,
  enable2FA: false,
  enableNotifications: true,
  enableModelMetrics: true,
  enableExperiments: true,
} as const;

// Error Messages
export const ERROR_MESSAGES = {
  validation: {
    network: 'Network error occurred. Please check your connection.',
    unauthorized: 'You are not authorized to perform this action.',
    serverError: 'Server error occurred. Please try again later.',
    validation: 'Please check your input and try again.',
    notFound: 'The requested resource was not found.',
    badRequest: 'Invalid request. Please check your input.',
    auth: {
      invalidCredentials: 'Invalid email or password.',
      sessionExpired: 'Your session has expired. Please log in again.',
    },
    required: 'This field is required.',
    invalidEmail: 'Please enter a valid email address.',
    passwordMismatch: 'Passwords do not match.',
    weakPassword: 'Password is too weak.',
  },
} as const;

// API Endpoints
export const API_ENDPOINTS = {
  auth: {
    login: '/auth/login',
    register: '/auth/register',
    logout: '/auth/logout',
    refresh: '/auth/refresh',
    profile: '/auth/profile',
  },
  models: {
    list: '/models',
    create: '/models',
    get: (id: string) => `/models/${id}`,
    update: (id: string) => `/models/${id}`,
    delete: (id: string) => `/models/${id}`,
    metrics: (id: string) => `/models/${id}/metrics`,
  },
  experiments: {
    list: '/experiments',
    create: '/experiments',
    get: (id: string) => `/experiments/${id}`,
    update: (id: string) => `/experiments/${id}`,
    delete: (id: string) => `/experiments/${id}`,
  },
  settings: {
    get: '/settings',
    update: '/settings',
  },
  profile: {
    get: '/profile',
    update: '/profile',
    changePassword: '/profile/password',
  },
} as const;
