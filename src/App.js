import React from 'react';
import { Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider } from '@mui/material/styles';
import theme from './theme';
import Navigation from './components/Navigation';
import Login from './components/auth/Login';
import Register from './components/auth/Register';
import Dashboard from './components/Dashboard';
import Experiments from './components/experiments/Experiments';
import ModelTraining from './components/model/ModelTraining';
import ModelEvaluation from './components/model/ModelEvaluation';
import ModelDeployment from './components/model/ModelDeployment';
import ModelMonitoring from './components/model/ModelMonitoring';
import Profile from './components/profile/Profile';
import PrivateRoute from './components/auth/PrivateRoute';
import { useSelector } from 'react-redux';
import { AuthProvider } from './context/AuthContext';
import ErrorBoundary from './components/common/ErrorBoundary';

function App() {
  const { token } = useSelector(state => state.auth);

  return (
    <AuthProvider>
      <ThemeProvider theme={theme}>
        <ErrorBoundary>
          <div className="App">
            <Navigation />
            <Routes>
              <Route path="/login" element={
                token ? <Navigate to="/dashboard" replace /> : <Login />
              } />
              <Route path="/register" element={
                token ? <Navigate to="/dashboard" replace /> : <Register />
              } />
              <Route
                path="/dashboard"
                element={
                  <PrivateRoute>
                    <Dashboard />
                  </PrivateRoute>
                }
              />
              <Route
                path="/experiments"
                element={
                  <PrivateRoute>
                    <Experiments />
                  </PrivateRoute>
                }
              />
              <Route
                path="/model/training"
                element={
                  <PrivateRoute>
                    <ModelTraining />
                  </PrivateRoute>
                }
              />
              <Route
                path="/model/evaluation"
                element={
                  <PrivateRoute>
                    <ModelEvaluation />
                  </PrivateRoute>
                }
              />
              <Route
                path="/model/deployment"
                element={
                  <PrivateRoute>
                    <ModelDeployment />
                  </PrivateRoute>
                }
              />
              <Route
                path="/model/monitoring"
                element={
                  <PrivateRoute>
                    <ModelMonitoring />
                  </PrivateRoute>
                }
              />
              <Route
                path="/profile"
                element={
                  <PrivateRoute>
                    <Profile />
                  </PrivateRoute>
                }
              />
              <Route path="/" element={<Navigate to="/dashboard" replace />} />
            </Routes>
          </div>
        </ErrorBoundary>
      </ThemeProvider>
    </AuthProvider>
  );
}

export default App; 