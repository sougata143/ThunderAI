import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const startTraining = createAsyncThunk(
  'training/startTraining',
  async (config, { rejectWithValue }) => {
    try {
      console.log('Starting training with config:', config);
      const response = await axios.post(`${API_BASE_URL}/models/train`, {
        model_type: config.modelType,
        batch_size: config.batchSize,
        epochs: config.epochs,
        learning_rate: config.learningRate,
        optimizer: config.optimizer,
        loss: config.loss,
        metrics: config.metrics
      });
      console.log('Training response:', response.data);
      return response.data;
    } catch (err) {
      console.error('Training error:', err.response?.data || err.message);
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          err.message ||
                          'Failed to start training';
      return rejectWithValue(errorMessage);
    }
  }
);

export const stopTraining = createAsyncThunk(
  'training/stopTraining',
  async (modelId, { rejectWithValue }) => {
    try {
      const response = await axios.post(`${API_BASE_URL}/models/${modelId}/stop`);
      return response.data;
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 
                          err.response?.data?.message || 
                          err.message ||
                          'Failed to stop training';
      return rejectWithValue(errorMessage);
    }
  }
);

const trainingSlice = createSlice({
  name: 'training',
  initialState: {
    status: 'idle',
    modelId: null,
    metrics: [],
    progress: 0,
    error: null
  },
  reducers: {
    updateMetrics: (state, action) => {
      state.metrics = action.payload;
    },
    updateProgress: (state, action) => {
      state.progress = action.payload;
    },
    setModelId: (state, action) => {
      state.modelId = action.payload;
    }
  },
  extraReducers: (builder) => {
    builder
      .addCase(startTraining.pending, (state) => {
        state.status = 'training';
        state.error = null;
      })
      .addCase(startTraining.fulfilled, (state, action) => {
        state.status = 'training';
        state.modelId = action.payload.model_id;
        state.error = null;
      })
      .addCase(startTraining.rejected, (state, action) => {
        state.status = 'failed';
        state.error = action.payload || 'Training failed';
        console.error('Training failed:', action.payload);
      })
      .addCase(stopTraining.pending, (state) => {
        state.status = 'stopping';
      })
      .addCase(stopTraining.fulfilled, (state) => {
        state.status = 'completed';
        state.error = null;
      })
      .addCase(stopTraining.rejected, (state, action) => {
        state.error = action.payload || 'Failed to stop training';
      });
  },
});

export const { updateMetrics, updateProgress, setModelId } = trainingSlice.actions;
export default trainingSlice.reducer; 