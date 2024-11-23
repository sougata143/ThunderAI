import React from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  PlayArrow as PlayArrowIcon,
  Analytics as AnalyticsIcon,
} from '@mui/icons-material';

export const ModelCard = ({ model, onDelete }) => {
  const navigate = useNavigate();

  // Debug log the received model
  console.log('ModelCard received model:', model);

  // Validate model data
  if (!model || !model._id) {
    console.error('ModelCard: Invalid model data:', model);
    return null;
  }

  // Ensure model._id is a string
  const modelId = typeof model._id === 'string' ? model._id : String(model._id);

  // Debug log the extracted modelId
  console.log('ModelCard: Using modelId:', modelId);

  // Validate modelId
  if (!modelId || modelId === 'undefined' || modelId === 'null') {
    console.error('ModelCard: Invalid model ID:', { modelId, originalId: model._id });
    return null;
  }

  const handleDelete = (e) => {
    e.preventDefault();
    e.stopPropagation();
    
    console.log('ModelCard: Delete clicked for model:', { modelId, model });
    
    if (!modelId || modelId === 'undefined' || modelId === 'null') {
      console.error('ModelCard: Cannot delete - invalid model ID:', modelId);
      return;
    }

    if (typeof onDelete !== 'function') {
      console.error('ModelCard: onDelete is not a function:', onDelete);
      return;
    }

    try {
      onDelete(modelId);
    } catch (error) {
      console.error('ModelCard: Error in delete handler:', error);
    }
  };

  const handleGenerate = (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigate(`/llm/generate/${modelId}`);
  };

  const handleMetrics = (e) => {
    e.preventDefault();
    e.stopPropagation();
    navigate(`/llm/metrics/${modelId}`);
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Typography variant="h5" component="div" gutterBottom>
          {model.name || 'Unnamed Model'}
        </Typography>
        <Typography variant="body2" color="text.secondary" paragraph>
          {model.description || 'No description available'}
        </Typography>
        <Box sx={{ mt: 2 }}>
          <Chip
            label={`Type: ${model.model_type || 'Unknown'}`}
            size="small"
            sx={{ mr: 1, mb: 1 }}
          />
          <Chip
            label={`Version: ${model.version || '1.0'}`}
            size="small"
            sx={{ mr: 1, mb: 1 }}
          />
          <Chip
            label={`ID: ${modelId}`}
            size="small"
            sx={{ mb: 1 }}
          />
        </Box>
      </CardContent>
      <CardActions sx={{ justifyContent: 'flex-end', p: 2 }}>
        <Tooltip title="Delete Model">
          <IconButton
            size="small"
            color="error"
            onClick={handleDelete}
            data-model-id={modelId}
          >
            <DeleteIcon />
          </IconButton>
        </Tooltip>
        <Tooltip title="View Metrics">
          <IconButton 
            size="small" 
            color="primary" 
            onClick={handleMetrics}
          >
            <AnalyticsIcon />
          </IconButton>
        </Tooltip>
        <Button
          variant="contained"
          color="primary"
          startIcon={<PlayArrowIcon />}
          onClick={handleGenerate}
        >
          Generate
        </Button>
      </CardActions>
    </Card>
  );
};
