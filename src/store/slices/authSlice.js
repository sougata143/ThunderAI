import { createSlice, createAsyncThunk } from '@reduxjs/toolkit';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000/api/v1';

export const register = createAsyncThunk(
    'auth/register',
    async (credentials, { rejectWithValue }) => {
        try {
            const response = await fetch(`${API_URL}/register`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(credentials),
            });

            const data = await response.json();

            if (!response.ok) {
                return rejectWithValue(data.detail || 'Registration failed');
            }

            return data;
        } catch (err) {
            console.error('Registration error:', err);
            return rejectWithValue('Network error occurred');
        }
    }
);

export const login = createAsyncThunk(
    'auth/login',
    async (credentials, { rejectWithValue }) => {
        try {
            const formData = new FormData();
            formData.append('username', credentials.email);
            formData.append('password', credentials.password);

            const response = await fetch(`${API_URL}/token`, {
                method: 'POST',
                body: formData,
            });

            const data = await response.json();

            if (!response.ok) {
                return rejectWithValue(data.detail || 'Login failed');
            }

            localStorage.setItem('token', data.access_token);
            localStorage.setItem('user', JSON.stringify(data.user));

            return data;
        } catch (error) {
            console.error('Login error:', error);
            return rejectWithValue('Network error occurred');
        }
    }
);

export const loginAsGuest = createAsyncThunk(
    'auth/loginAsGuest',
    async (_, { rejectWithValue }) => {
        try {
            const response = await fetch(`${API_URL}/guest-token`, {
                method: 'POST',
            });

            const data = await response.json();

            if (!response.ok) {
                return rejectWithValue(data.detail || 'Guest login failed');
            }

            return {
                ...data,
                isGuest: true
            };
        } catch (error) {
            return rejectWithValue('Guest login failed');
        }
    }
);

const authSlice = createSlice({
    name: 'auth',
    initialState: {
        user: JSON.parse(localStorage.getItem('user')) || null,
        token: localStorage.getItem('token') || null,
        loading: false,
        error: null,
        isGuest: false
    },
    reducers: {
        logout: (state) => {
            state.user = null;
            state.token = null;
            state.isGuest = false;
            localStorage.removeItem('token');
            localStorage.removeItem('user');
        },
        clearError: (state) => {
            state.error = null;
        },
    },
    extraReducers: (builder) => {
        builder
            .addCase(register.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(register.fulfilled, (state, action) => {
                state.loading = false;
                state.error = null;
            })
            .addCase(register.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            .addCase(login.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(login.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                state.token = action.payload.access_token;
                state.error = null;
            })
            .addCase(login.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            })
            .addCase(loginAsGuest.pending, (state) => {
                state.loading = true;
                state.error = null;
            })
            .addCase(loginAsGuest.fulfilled, (state, action) => {
                state.loading = false;
                state.user = action.payload.user;
                state.token = action.payload.access_token;
                state.isGuest = true;
                state.error = null;
            })
            .addCase(loginAsGuest.rejected, (state, action) => {
                state.loading = false;
                state.error = action.payload;
            });
    },
});

export const { logout, clearError } = authSlice.actions;
export default authSlice.reducer; 