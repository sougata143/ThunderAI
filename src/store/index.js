import { configureStore } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import trainingReducer from './slices/trainingSlice';

export const store = configureStore({
    reducer: {
        auth: authReducer,
        training: trainingReducer,
    },
});

export default store; 