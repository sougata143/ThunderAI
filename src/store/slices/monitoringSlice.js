import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  metrics: [],
  alerts: [],
  loading: false,
  error: null,
};

const monitoringSlice = createSlice({
  name: 'monitoring',
  initialState,
  reducers: {
    fetchMetricsStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchMetricsSuccess: (state, action) => {
      state.loading = false;
      state.metrics = action.payload;
    },
    fetchMetricsFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    addAlert: (state, action) => {
      state.alerts.push(action.payload);
    },
    clearAlerts: (state) => {
      state.alerts = [];
    },
  },
});

export const {
  fetchMetricsStart,
  fetchMetricsSuccess,
  fetchMetricsFailure,
  addAlert,
  clearAlerts,
} = monitoringSlice.actions;

export default monitoringSlice.reducer; 