import axiosInstance from './axiosConfig';

export const experimentService = {
  async getExperiments() {
    try {
      const response = await axiosInstance.get('/experiments');
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async startExperiment(id) {
    try {
      const response = await axiosInstance.post(`/experiments/${id}/start`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async stopExperiment(id) {
    try {
      const response = await axiosInstance.post(`/experiments/${id}/stop`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async deleteExperiment(id) {
    try {
      const response = await axiosInstance.delete(`/experiments/${id}`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  },

  async getExperimentMetrics(id) {
    try {
      const response = await axiosInstance.get(`/experiments/${id}/metrics`);
      return response.data;
    } catch (error) {
      throw error.response?.data || error.message;
    }
  }
}; 