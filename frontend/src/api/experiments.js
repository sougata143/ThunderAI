// Using relative URLs since we're proxying through Vite
const API_BASE = '/api/v1';

export const fetchExperiments = async () => {
  try {
    const response = await fetch(`${API_BASE}/experiments`);
    if (!response.ok) {
      throw new Error(`Failed to fetch experiments: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching experiments:', error);
    throw error;
  }
};

export const createExperiment = async (experimentData) => {
  try {
    const response = await fetch(`${API_BASE}/experiments`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(experimentData),
    });
    if (!response.ok) {
      throw new Error(`Failed to create experiment: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error creating experiment:', error);
    throw error;
  }
};

export const getExperiment = async (id) => {
  try {
    const response = await fetch(`${API_BASE}/experiments/${id}`);
    if (!response.ok) {
      throw new Error(`Failed to fetch experiment: ${response.statusText}`);
    }
    const data = await response.json();
    return data;
  } catch (error) {
    console.error('Error fetching experiment:', error);
    throw error;
  }
};
