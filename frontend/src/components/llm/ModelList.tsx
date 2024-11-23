import React, { useState, useEffect } from 'react';
import {
  Box,
  Button,
  CircularProgress,
  Grid,
  Typography,
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';
import { api } from '../../services/api';
import { ModelCard } from './ModelCard';
import { CreateModelDialog } from './CreateModelDialog';
import { toast } from 'react-hot-toast';

interface Model {
  id: string;
  name: string;
  description: string;
  architecture: string;
  status: string;
  created_at: string;
  metrics?: {
    perplexity: number;
    bleu_score?: number;
    accuracy?: number;
    loss: number;
  };
}

export const ModelList: React.FC = () => {
  const [models, setModels] = useState<Model[]>([]);
  const [loading, setLoading] = useState(true);
  const [createDialogOpen, setCreateDialogOpen] = useState(false);
  const [creatingModel, setCreatingModel] = useState(false);

  const fetchModels = async () => {
    try {
      const response = await api.get('/llm/models');
      setModels(response.data);
    } catch (error) {
      toast.error('Failed to fetch models');
      console.error('Error fetching models:', error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const handleCreateModel = async (modelData: any) => {
    setCreatingModel(true);
    try {
      // Create the model
      const createResponse = await api.post('/llm/models', modelData);
      const newModelId = createResponse.data;
      
      if (newModelId) {
        // Wait a bit for the model to be fully created
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Fetch the newly created model
        const modelResponse = await api.get(`/llm/models/${newModelId}`);
        const newModel = modelResponse.data;
        
        setModels(prevModels => [...prevModels, newModel]);
        toast.success('Model created successfully');
        setCreateDialogOpen(false);
      }
    } catch (error: any) {
      console.error('Error creating model:', error);
      toast.error(error.response?.data?.detail || 'Failed to create model');
    } finally {
      setCreatingModel(false);
    }
  };

  const handleDeleteModel = async (modelId: string) => {
    try {
      await api.delete(`/llm/models/${modelId}`);
      setModels(prevModels => prevModels.filter(model => model.id !== modelId));
      toast.success('Model deleted successfully');
    } catch (error: any) {
      toast.error(error.response?.data?.detail || 'Failed to delete model');
      console.error('Error deleting model:', error);
    }
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
        <Typography variant="h5" component="h1">
          Language Models
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => setCreateDialogOpen(true)}
          disabled={creatingModel}
        >
          {creatingModel ? 'Creating...' : 'Create Model'}
        </Button>
      </Box>

      <Grid container spacing={3}>
        {models.map((model) => (
          <Grid item xs={12} sm={6} md={4} key={model.id}>
            <ModelCard
              model={model}
              onDelete={() => handleDeleteModel(model.id)}
            />
          </Grid>
        ))}
        {models.length === 0 && !loading && (
          <Grid item xs={12}>
            <Box textAlign="center" py={4}>
              <Typography color="textSecondary">
                No models found. Create your first model to get started.
              </Typography>
            </Box>
          </Grid>
        )}
      </Grid>

      <CreateModelDialog
        open={createDialogOpen}
        onClose={() => !creatingModel && setCreateDialogOpen(false)}
        onSubmit={handleCreateModel}
      />
    </Box>
  );
};
