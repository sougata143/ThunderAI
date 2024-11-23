import axios from 'axios';

const API_URL = 'http://localhost:9000/api/v1';

export const fetchUsers = async () => {
  const response = await axios.get(`${API_URL}/users/`);
  return response.data;
};

export const createUser = async (userData) => {
  const response = await axios.post(`${API_URL}/users/`, userData);
  return response.data;
};

export const updateUser = async (userId, userData) => {
  const response = await axios.put(`${API_URL}/users/${userId}`, userData);
  return response.data;
};

export const deleteUser = async (userId) => {
  const response = await axios.delete(`${API_URL}/users/${userId}`);
  return response.data;
};
