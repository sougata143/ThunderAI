import React, { useState, useEffect } from 'react';
import { Container, Grid, Typography, Button, Box, Alert } from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import { CreateModelDialog } from './CreateModelDialog';
import { ModelCard } from './ModelCard';
import { api } from '../../services/api';

export const ModelList = () => {
  const { user } = useAuth();  // We don't need token directly, api will handle it
  const [models, setModels] = useState([]);
  const [open, setOpen] = useState(false);
  const [error, setError] = useState('');

  const fetchModels = async () => {
    try {
      console.log('Fetching models...');
      const response = await api.get('/llm/models');
      
      // Debug log raw data
      console.log('Raw models data:', response.data);
      
      // Ensure model IDs are strings and validate data structure
      const processedModels = response.data.map(model => {
        if (!model || !model._id) {
          console.error('Invalid model data:', model);
          return null;
        }
        return {
          ...model,
          _id: model._id?.toString() || model._id
        };
      }).filter(model => model !== null); // Remove any invalid models
      
      // Debug log processed models
      console.log('Processed models:', processedModels);
      
      setModels(processedModels);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError(err.response?.data?.detail || err.message);
    }
  };

  useEffect(() => {
    if (user) {  // Only fetch if user is logged in
      fetchModels();
    }
  }, [user]);  // Depend on user instead of token

  const handleCreateModel = async (modelData) => {
    try {
      console.log('Creating model with data:', modelData);
      const response = await api.post('/llm/models', modelData);
      console.log('Model created:', response.data);
      await fetchModels();  // Refresh the list
      setOpen(false);
    } catch (err) {
      console.error('Error creating model:', err);
      setError(err.response?.data?.detail || err.message);
    }
  };

  const handleDeleteModel = async (modelId) => {
    try {
      console.log('Deleting model:', modelId);
      await api.delete(`/llm/models/${modelId}`);
      console.log('Model deleted');
      await fetchModels();  // Refresh the list
    } catch (err) {
      console.error('Error deleting model:', err);
      setError(err.response?.data?.detail || err.message);
    }
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4" component="h1">
          LLM Models
        </Typography>
        <Button
          variant="contained"
          startIcon={<AddIcon />}
          onClick={() => setOpen(true)}
        >
          Add Model
        </Button>
      </Box>

      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
          {error}
        </Alert>
      )}

      <Grid container spacing={3}>
        {models.map((model) => {
          // Skip invalid models
          if (!model || !model._id) {
            console.error('Invalid model data:', model);
            return null;
          }

          // Ensure model ID is a string
          const modelId = typeof model._id === 'object' && model._id !== null
            ? model._id.toString()
            : model._id;

          if (!modelId || modelId === 'undefined') {
            console.error('Invalid model ID:', { model, modelId });
            return null;
          }

          return (
            <Grid item xs={12} md={6} lg={4} key={modelId}>
              <ModelCard 
                model={{ ...model, _id: modelId }}
                onDelete={() => handleDeleteModel(modelId)}
              />
            </Grid>
          );
        })}
      </Grid>

      <CreateModelDialog
        open={open}
        onClose={() => setOpen(false)}
        onSubmit={handleCreateModel}
        error={error}
      />
    </Container>
  );
};
