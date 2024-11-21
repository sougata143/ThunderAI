import { configureStore } from '@reduxjs/toolkit';
import authReducer from './slices/authSlice';
import modelReducer from './slices/modelSlice';
import monitoringReducer from './slices/monitoringSlice';

const store = configureStore({
  reducer: {
    auth: authReducer,
    models: modelReducer,
    monitoring: monitoringReducer,
  },
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({
      serializableCheck: false,
    }),
});

export default store; 