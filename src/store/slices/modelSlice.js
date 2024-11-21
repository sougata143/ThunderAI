import { createSlice } from '@reduxjs/toolkit';

const initialState = {
  models: [],
  selectedModel: null,
  loading: false,
  error: null,
};

const modelSlice = createSlice({
  name: 'models',
  initialState,
  reducers: {
    fetchModelsStart: (state) => {
      state.loading = true;
      state.error = null;
    },
    fetchModelsSuccess: (state, action) => {
      state.loading = false;
      state.models = action.payload;
    },
    fetchModelsFailure: (state, action) => {
      state.loading = false;
      state.error = action.payload;
    },
    selectModel: (state, action) => {
      state.selectedModel = action.payload;
    },
  },
});

export const {
  fetchModelsStart,
  fetchModelsSuccess,
  fetchModelsFailure,
  selectModel,
} = modelSlice.actions;

export default modelSlice.reducer; 