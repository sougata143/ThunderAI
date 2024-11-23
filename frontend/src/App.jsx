import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Layout from './components/layout/Layout';
import { Dashboard } from './components/dashboard/Dashboard';
import { ExperimentDetails } from './components/experiments/ExperimentDetails';
import { NewExperiment } from './components/experiments/NewExperiment';
import { UsersList } from './components/users/UsersList';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ModelList } from './components/llm/ModelList';
import { TextGeneration } from './components/llm/TextGeneration';
import { ModelMetrics } from './components/llm/ModelMetrics';
import { Settings } from './pages/Settings';
import { Profile } from './components/profile/Profile';
import { Login } from './components/auth/Login';
import { Register } from './components/auth/Register';
import { PrivateRoute } from './components/auth/PrivateRoute';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: '#1976d2',
    },
    secondary: {
      main: '#dc004e',
    },
    background: {
      default: '#f5f5f5',
    },
  },
  typography: {
    fontFamily: '"Roboto", "Helvetica", "Arial", sans-serif',
  },
});

const AppRoutes = () => {
  const { user, loading } = useAuth();

  if (loading) {
    return null;
  }

  return (
    <Routes>
      <Route path="/login" element={<Login />} />
      <Route path="/register" element={<Register />} />
      
      <Route element={<Layout />}>
        <Route path="/" element={
          <PrivateRoute>
            <Dashboard />
          </PrivateRoute>
        } />
        <Route path="/dashboard" element={
          <PrivateRoute>
            <Dashboard />
          </PrivateRoute>
        } />
        <Route path="/experiments/new" element={
          <PrivateRoute>
            <NewExperiment />
          </PrivateRoute>
        } />
        <Route path="/experiments/:id" element={
          <PrivateRoute>
            <ExperimentDetails />
          </PrivateRoute>
        } />
        <Route path="/users" element={
          <PrivateRoute>
            <UsersList />
          </PrivateRoute>
        } />
        <Route path="/llm" element={
          <PrivateRoute>
            <ModelList />
          </PrivateRoute>
        } />
        <Route path="/llm/generate/:modelId" element={
          <PrivateRoute>
            <TextGeneration />
          </PrivateRoute>
        } />
        <Route path="/llm/metrics/:modelId" element={
          <PrivateRoute>
            <ModelMetrics />
          </PrivateRoute>
        } />
        <Route path="/settings" element={
          <PrivateRoute>
            <Settings />
          </PrivateRoute>
        } />
        <Route path="/profile" element={
          <PrivateRoute>
            <Profile />
          </PrivateRoute>
        } />
      </Route>
      
      <Route path="*" element={<Navigate to={user ? "/" : "/login"} replace />} />
    </Routes>
  );
};

function App() {
  return (
    <ThemeProvider theme={theme}>
      <CssBaseline />
      <AuthProvider>
        <Router>
          <AppRoutes />
        </Router>
      </AuthProvider>
    </ThemeProvider>
  );
}

export default App;
