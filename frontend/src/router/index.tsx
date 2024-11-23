import React from 'react';
import { createBrowserRouter, Navigate } from 'react-router-dom';
import { Layout } from '../components/Layout';
import { Login } from '../components/auth/Login';
import { Register } from '../components/auth/Register';
import { Dashboard } from '../components/dashboard/Dashboard';
import { ModelList } from '../components/llm/ModelList';
import { TextGeneration } from '../components/llm/TextGeneration';
import { ModelMetrics } from '../components/llm/ModelMetrics';
import { ProtectedRoute } from './ProtectedRoute';

export const router = createBrowserRouter([
  {
    path: '/',
    element: <Layout />,
    children: [
      {
        index: true,
        element: <Navigate to="/dashboard" replace />,
      },
      {
        path: 'login',
        element: <Login />,
      },
      {
        path: 'register',
        element: <Register />,
      },
      {
        path: 'dashboard',
        element: (
          <ProtectedRoute>
            <Dashboard />
          </ProtectedRoute>
        ),
      },
      {
        path: 'llm',
        element: <ProtectedRoute />,
        children: [
          {
            index: true,
            element: <ModelList />,
          },
          {
            path: 'generate/:modelId',
            element: <TextGeneration />,
          },
          {
            path: 'metrics/:modelId',
            element: <ModelMetrics />,
          },
        ],
      },
    ],
  },
]);
