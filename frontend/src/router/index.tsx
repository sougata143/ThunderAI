import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import Layout from '../components/Layout/Layout';
import Login from '../components/auth/Login';
import Register from '../components/auth/Register';
import { Dashboard } from '../components/dashboard/Dashboard';
import ModelList from '../components/llm/ModelList';
import { TextGenerationWithErrorBoundary } from '../components/llm/TextGeneration';
import { ModelMetrics } from '../components/llm/ModelMetrics';
import Settings from '../components/settings/Settings';
import Profile from '../components/profile/Profile';
import { Experiments } from '../components/experiments/Experiments';
import { ProtectedRoute } from './ProtectedRoute';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Navigate to="/app/dashboard" replace />
  },
  {
    path: '/login',
    element: <Login />
  },
  {
    path: '/register',
    element: <Register />
  },
  {
    path: '/app',
    element: <ProtectedRoute><Layout /></ProtectedRoute>,
    children: [
      {
        index: true,
        element: <Navigate to="/app/dashboard" replace />
      },
      {
        path: 'dashboard',
        element: <Dashboard />
      },
      {
        path: 'models',
        element: <ModelList />
      },
      {
        path: 'llm/generate/:modelId',
        element: <TextGenerationWithErrorBoundary />
      },
      {
        path: 'llm/metrics/:modelId',
        element: <ModelMetrics />
      },
      {
        path: 'experiments',
        element: <Experiments />
      },
      {
        path: 'settings',
        element: <Settings />
      },
      {
        path: 'profile',
        element: <Profile />
      }
    ]
  }
]);
