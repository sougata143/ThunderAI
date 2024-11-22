import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000/api/v1';

// Create axios instance with default config
const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'application/json',
    },
});

// Add request interceptor to inject auth token
apiClient.interceptors.request.use(
    (config) => {
        const token = localStorage.getItem('token');
        if (token) {
            config.headers.Authorization = `Bearer ${token.replace('Bearer ', '')}`;
        }
        return config;
    },
    (error) => {
        return Promise.reject(new Error(error.response?.data?.detail || 'Request failed'));
    }
);

// Add response interceptor to handle auth errors
apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
        if (error.response?.status === 401 || error.response?.status === 403) {
            localStorage.removeItem('token');
            window.location.href = '/login';
        }
        const errorMessage = error.response?.data?.detail 
            ? (Array.isArray(error.response.data.detail) 
                ? error.response.data.detail.map(err => err.msg).join(', ')
                : error.response.data.detail)
            : 'An error occurred';
        return Promise.reject(new Error(errorMessage));
    }
);

export const modelService = {
    startTraining: async (config) => {
        try {
            // Format training config according to TrainingParams model
            const requestBody = {
                training_params: {
                    model_type: config.modelType,
                    optimizer: config.optimizer,
                    loss: config.loss,
                    metrics: Array.isArray(config.metrics) ? config.metrics : ['accuracy'],
                    validation_split: Number(config.validationSplit),
                    shuffle: Boolean(config.shuffle),
                    verbose: Number(config.verbose),
                    batch_size: Number(config.batchSize),
                    epochs: Number(config.epochs),
                    learning_rate: Number(config.learningRate)
                }
            };

            // Validate numeric fields
            const { training_params } = requestBody;
            if (isNaN(training_params.batch_size) || training_params.batch_size <= 0) {
                throw new Error('Invalid batch size');
            }
            if (isNaN(training_params.epochs) || training_params.epochs <= 0) {
                throw new Error('Invalid epochs');
            }
            if (isNaN(training_params.learning_rate) || training_params.learning_rate <= 0) {
                throw new Error('Invalid learning rate');
            }
            if (isNaN(training_params.validation_split) || 
                training_params.validation_split < 0 || 
                training_params.validation_split > 1) {
                throw new Error('Invalid validation split');
            }

            console.log('Request body:', requestBody);
            
            const response = await apiClient.post('/models/train', requestBody);
            
            return response.data;
        } catch (error) {
            console.error('Error starting training:', error);
            if (error.response?.data?.detail) {
                const detail = error.response.data.detail;
                if (Array.isArray(detail)) {
                    throw new Error(detail.map(err => err.msg).join(', '));
                } else {
                    throw new Error(detail);
                }
            }
            throw new Error(error.message || 'Failed to start training');
        }
    },

    startDashboardTraining: async (dashboardConfig) => {
        try {
            // Format training config for dashboard
            const requestBody = {
                training_params: {
                    model_type: dashboardConfig.modelType || 'bert',
                    optimizer: dashboardConfig.optimizer || 'adam',
                    loss: dashboardConfig.loss || 'categorical_crossentropy',
                    metrics: ['accuracy'],
                    validation_split: 0.2,
                    shuffle: true,
                    verbose: 1,
                    batch_size: Number(dashboardConfig.batchSize) || 32,
                    epochs: Number(dashboardConfig.epochs) || 10,
                    learning_rate: Number(dashboardConfig.learningRate) || 0.001
                }
            };

            console.log('Dashboard training request:', requestBody);
            
            const response = await apiClient.post('/models/train', requestBody);
            
            return response.data;
        } catch (error) {
            console.error('Error starting dashboard training:', error);
            if (error.response?.data?.detail) {
                const detail = error.response.data.detail;
                if (Array.isArray(detail)) {
                    throw new Error(detail.map(err => err.msg).join(', '));
                } else {
                    throw new Error(detail);
                }
            }
            throw new Error(error.message || 'Failed to start training');
        }
    },

    getExperiments: async () => {
        try {
            const response = await apiClient.get('/experiments');
            return response;
        } catch (error) {
            console.error('Error fetching experiments:', error.message);
            throw new Error(error.message || 'Failed to fetch experiments');
        }
    },

    stopTraining: async (modelId) => {
        try {
            const response = await apiClient.post(`/models/${modelId}/stop`);
            return response.data;
        } catch (error) {
            console.error('Error stopping training:', error);
            throw new Error(error.response?.data?.detail || 'Failed to stop training');
        }
    },
}; 