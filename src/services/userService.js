import axiosInstance from './axiosConfig';

export const userService = {
  async getProfile() {
    try {
      const response = await axiosInstance.get('/users/profile');
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async updateProfile(profileData) {
    try {
      const response = await axiosInstance.put('/users/profile', profileData);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async updatePassword(passwordData) {
    try {
      const response = await axiosInstance.put('/users/password', passwordData);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async updateSettings(settings) {
    try {
      const response = await axiosInstance.put('/users/settings', settings);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  }
}; 