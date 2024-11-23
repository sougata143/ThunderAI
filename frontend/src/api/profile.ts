import axios from 'axios';

export interface UserProfile {
  id: string;
  email: string;
  name: string;
  role: string;
  avatar?: string;
  organization?: string;
  jobTitle?: string;
  bio?: string;
  preferences: {
    emailNotifications: boolean;
    twoFactorEnabled: boolean;
    theme: 'light' | 'dark' | 'system';
    language: string;
  };
  stats: {
    modelsCreated: number;
    experimentsRun: number;
    totalTrainingHours: number;
    lastActive: string;
  };
  apiKeys: Array<{
    id: string;
    name: string;
    lastUsed: string;
    createdAt: string;
  }>;
}

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

export const profileApi = {
  getProfile: async (): Promise<UserProfile> => {
    try {
      const response = await axios.get(`${API_URL}/api/v1/profile`);
      return response.data;
    } catch (error) {
      console.error('Failed to fetch profile:', error);
      throw error;
    }
  },

  updateProfile: async (profile: Partial<UserProfile>): Promise<UserProfile> => {
    try {
      const response = await axios.put(`${API_URL}/api/v1/profile`, profile);
      return response.data;
    } catch (error) {
      console.error('Failed to update profile:', error);
      throw error;
    }
  },

  changePassword: async (data: { currentPassword: string; newPassword: string }): Promise<void> => {
    try {
      await axios.post(`${API_URL}/api/v1/profile/change-password`, data);
    } catch (error) {
      console.error('Failed to change password:', error);
      throw error;
    }
  },

  generateApiKey: async (name: string): Promise<{ key: string; id: string }> => {
    try {
      const response = await axios.post(`${API_URL}/api/v1/profile/api-keys`, { name });
      return response.data;
    } catch (error) {
      console.error('Failed to generate API key:', error);
      throw error;
    }
  },

  deleteApiKey: async (keyId: string): Promise<void> => {
    try {
      await axios.delete(`${API_URL}/api/v1/profile/api-keys/${keyId}`);
    } catch (error) {
      console.error('Failed to delete API key:', error);
      throw error;
    }
  },

  uploadAvatar: async (file: File): Promise<{ url: string }> => {
    try {
      const formData = new FormData();
      formData.append('avatar', file);
      const response = await axios.post(`${API_URL}/api/v1/profile/avatar`, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      });
      return response.data;
    } catch (error) {
      console.error('Failed to upload avatar:', error);
      throw error;
    }
  },

  enable2FA: async (): Promise<{ qrCode: string; secret: string }> => {
    try {
      const response = await axios.post(`${API_URL}/api/v1/profile/2fa/enable`);
      return response.data;
    } catch (error) {
      console.error('Failed to enable 2FA:', error);
      throw error;
    }
  },

  verify2FA: async (token: string): Promise<void> => {
    try {
      await axios.post(`${API_URL}/api/v1/profile/2fa/verify`, { token });
    } catch (error) {
      console.error('Failed to verify 2FA:', error);
      throw error;
    }
  },

  disable2FA: async (token: string): Promise<void> => {
    try {
      await axios.post(`${API_URL}/api/v1/profile/2fa/disable`, { token });
    } catch (error) {
      console.error('Failed to disable 2FA:', error);
      throw error;
    }
  },
};
