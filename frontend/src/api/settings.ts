import axios from 'axios';

interface SettingsState {
  modelSettings: {
    defaultModelType: string;
    maxBatchSize: number;
    trainingTimeout: number;
    useGPU: boolean;
  };
  apiSettings: {
    baseUrl: string;
    timeout: number;
    retryAttempts: number;
    enableCaching: boolean;
  };
  uiSettings: {
    darkMode: boolean;
    autoRefresh: boolean;
    refreshInterval: number;
    compactView: boolean;
  };
  notificationSettings: {
    emailNotifications: boolean;
    trainingComplete: boolean;
    errorAlerts: boolean;
    systemUpdates: boolean;
  };
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const settingsApi = {
  getSettings: async (): Promise<SettingsState> => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/settings`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch settings:', error);
      throw error;
    }
  },

  updateSettings: async (settings: SettingsState): Promise<SettingsState> => {
    try {
      const response = await axios.put(`${API_URL}/api/v1/settings`, settings);
      return response.data;
    } catch (error) {
      console.error('Failed to update settings:', error);
      throw error;
    }
  },

  resetSettings: async (): Promise<SettingsState> => {
    try {
      const response = await axios.post(`${API_URL}/api/v1/settings/reset`);
      return response.data;
    } catch (error) {
      console.error('Failed to reset settings:', error);
      throw error;
    }
  },
};
