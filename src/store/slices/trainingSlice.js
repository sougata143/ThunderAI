import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';
import { modelService } from '../../services/modelService';

export const startTraining = createAsyncThunk(
    'training/start',
    async (modelConfig, { rejectWithValue }) => {
        try {
            const response = await modelService.startTraining(modelConfig);
            return response;
        } catch (error) {
            return rejectWithValue(error);
        }
    }
);

export const stopTraining = createAsyncThunk(
    'training/stop',
    async (modelId, { rejectWithValue }) => {
        try {
            const response = await modelService.stopTraining(modelId);
            return response;
        } catch (error) {
            return rejectWithValue(error);
        }
    }
);

const trainingSlice = createSlice({
    name: 'training',
    initialState: {
        currentModel: null,
        status: 'idle',
        progress: 0,
        metrics: {
            loss: [],
            accuracy: []
        },
        error: null
    },
    reducers: {
        updateMetrics: (state, action) => {
            state.metrics = {
                ...state.metrics,
                ...action.payload
            };
        },
        updateProgress: (state, action) => {
            state.progress = action.payload;
        },
        setError: (state, action) => {
            state.error = action.payload;
            state.status = 'failed';
        },
        resetTraining: (state) => {
            state.currentModel = null;
            state.status = 'idle';
            state.progress = 0;
            state.metrics = {
                loss: [],
                accuracy: []
            };
            state.error = null;
        }
    },
    extraReducers: (builder) => {
        builder
            .addCase(startTraining.pending, (state) => {
                state.status = 'loading';
                state.error = null;
            })
            .addCase(startTraining.fulfilled, (state, action) => {
                state.status = 'training';
                state.currentModel = action.payload.modelId;
                state.error = null;
            })
            .addCase(startTraining.rejected, (state, action) => {
                state.status = 'failed';
                state.error = action.payload;
            })
            .addCase(stopTraining.fulfilled, (state) => {
                state.status = 'stopped';
            });
    }
});

export const { updateMetrics, updateProgress, setError, resetTraining } = trainingSlice.actions;
export default trainingSlice.reducer; 