import axios from 'axios';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const modelService = {
    async startTraining(modelConfig) {
        try {
            const response = await axios.post(`${API_URL}/models/train`, modelConfig, {
                headers: {
                    'Content-Type': 'application/json',
                }
            });
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async stopTraining(modelId) {
        try {
            const response = await axios.post(`${API_URL}/models/${modelId}/stop`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async getTrainingStatus(modelId) {
        try {
            const response = await axios.get(`${API_URL}/models/${modelId}/status`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async getExperiments() {
        try {
            const response = await axios.get(`${API_URL}/experiments`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async startExperiment(experimentId) {
        try {
            const response = await axios.post(`${API_URL}/experiments/${experimentId}/start`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async stopExperiment(experimentId) {
        try {
            const response = await axios.post(`${API_URL}/experiments/${experimentId}/stop`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async deleteExperiment(experimentId) {
        try {
            const response = await axios.delete(`${API_URL}/experiments/${experimentId}`);
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    },

    async exportExperimentResults(experimentId) {
        try {
            const response = await axios.get(
                `${API_URL}/experiments/${experimentId}/export`,
                { responseType: 'blob' }
            );
            
            // Create and trigger download
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', `experiment_${experimentId}_results.csv`);
            document.body.appendChild(link);
            link.click();
            link.remove();
            
            return response.data;
        } catch (error) {
            throw error.response?.data || error.message;
        }
    }
}; 