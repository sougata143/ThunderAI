import { instance as axios } from './axios';
import { API_ENDPOINTS } from '../config';

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
    pushNotifications: boolean;
    notifyOnError: boolean;
    notifyOnCompletion: boolean;
  };
}

export const settingsApi = {
  getSettings: async (): Promise<SettingsState> => {
    try {
      const response = await axios.get(API_ENDPOINTS.settings.get);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch settings:', error);
      throw error;
    }
  },

  updateSettings: async (settings: SettingsState): Promise<SettingsState> => {
    try {
      const response = await axios.put(API_ENDPOINTS.settings.update, settings);
      return response.data;
    } catch (error) {
      console.error('Failed to update settings:', error);
      throw error;
    }
  },

  resetSettings: async (): Promise<SettingsState> => {
    try {
      const response = await axios.post(`${API_ENDPOINTS.settings.update}/reset`);
      return response.data;
    } catch (error) {
      console.error('Failed to reset settings:', error);
      throw error;
    }
  },
};
