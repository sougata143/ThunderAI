import axios from 'axios';

class SharingService {
  static async shareVisualization(visualizationData) {
    try {
      // Generate a unique sharing ID
      const response = await axios.post('/api/v1/share/visualization', visualizationData);
      return response.data.shareUrl;
    } catch (error) {
      throw new Error('Failed to share visualization');
    }
  }

  static async shareDashboard(dashboardConfig) {
    try {
      // Save dashboard configuration and generate sharing link
      const response = await axios.post('/api/v1/share/dashboard', dashboardConfig);
      return response.data.shareUrl;
    } catch (error) {
      throw new Error('Failed to share dashboard');
    }
  }

  static async getSharedContent(shareId) {
    try {
      const response = await axios.get(`/api/v1/share/${shareId}`);
      return response.data;
    } catch (error) {
      throw new Error('Failed to fetch shared content');
    }
  }
}

export default SharingService; 