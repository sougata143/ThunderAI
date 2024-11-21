import React from 'react';
import { Routes, Route, Navigate, useLocation, useNavigate } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './components/Home';
import Model from './components/Model';
import Dashboard from './components/Dashboard';
import Experiments from './components/Experiments';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import { ThemeProvider, createTheme, CircularProgress, Box } from '@mui/material';
import './App.css';
import ErrorBoundary from './components/common/ErrorBoundary';
import { useSelector } from 'react-redux';
import { useEffect } from 'react';
import { setupAxiosInterceptors } from './services/authMiddleware';
import axios from 'axios';
import Profile from './components/Profile';

const theme = createTheme({
  palette: {
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
  },
});

function PrivateRoute({ children, requiresAuth = true, allowGuest = true }) {
  const { user, isGuest, loading } = useSelector(state => state.auth);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (loading) return;
    
    if (!user) {
      navigate('/login', { replace: true });
      return;
    }

    if (isGuest && !allowGuest) {
      navigate('/dashboard', { replace: true });
    }
  }, [user, isGuest, allowGuest, loading, navigate]);

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
        <CircularProgress />
      </Box>
    );
  }

  return children;
}

function App() {
  useEffect(() => {
    setupAxiosInterceptors(axios);
  }, []);

  return (
    <ThemeProvider theme={theme}>
      <ErrorBoundary>
        <div className="App">
          <Navigation />
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route path="/register" element={<Register />} />
            <Route
              path="/dashboard"
              element={
                <PrivateRoute allowGuest={true}>
                  <Dashboard />
                </PrivateRoute>
              }
            />
            <Route
              path="/experiments"
              element={
                <PrivateRoute allowGuest={false}>
                  <Experiments />
                </PrivateRoute>
              }
            />
            <Route
              path="/model"
              element={
                <PrivateRoute allowGuest={false}>
                  <Model />
                </PrivateRoute>
              }
            />
            <Route
              path="/profile"
              element={
                <PrivateRoute allowGuest={false}>
                  <Profile />
                </PrivateRoute>
              }
            />
            <Route path="/" element={<Navigate to="/dashboard" replace />} />
          </Routes>
        </div>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App; 