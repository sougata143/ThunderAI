import React from 'react';
import { Routes, Route, Navigate, useLocation } from 'react-router-dom';
import Navigation from './components/Navigation';
import Home from './components/Home';
import Model from './components/Model';
import Dashboard from './components/Dashboard';
import Experiments from './components/Experiments';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import { AuthProvider, useAuth } from './context/AuthContext';
import { ThemeProvider, createTheme, CircularProgress } from '@mui/material';
import './App.css';
import ErrorBoundary from './components/common/ErrorBoundary';
import { useSelector } from 'react-redux';

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

// Updated PrivateRoute component for React Router v6
function PrivateRoute({ children, requiresAuth = true, allowGuest = true }) {
  const { isAuthenticated, isGuest, loading } = useSelector(state => state.auth);
  const location = useLocation();

  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', marginTop: '2rem' }}>
        <CircularProgress />
      </div>
    );
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (isGuest && !allowGuest) {
    return <Navigate to="/dashboard" replace />;
  }

  return children;
}

function App() {
  return (
    <ThemeProvider theme={theme}>
      <ErrorBoundary>
        <AuthProvider>
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
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </AuthProvider>
      </ErrorBoundary>
    </ThemeProvider>
  );
}

export default App; 