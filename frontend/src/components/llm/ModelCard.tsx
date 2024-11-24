import React, { useState } from 'react';
import {
  Card,
  CardContent,
  Typography,
  IconButton,
  Box,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  LinearProgress,
  Tooltip,
  Grid,
} from '@mui/material';
import {
  Delete as DeleteIcon,
  PlayArrow as PlayIcon,
  Assessment as AssessmentIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';
import { formatDistanceToNow } from 'date-fns';

interface Model {
  _id: string;
  name: string;
  description?: string;
  model_type: string;
  version: string;
  parameters?: Record<string, any>;
  metadata?: Record<string, any>;
  created_at: string;
  updated_at?: string;
  is_active: boolean;
}

interface ModelCardProps {
  model: Model;
  onDelete: (id: string) => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model, onDelete }) => {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const navigate = useNavigate();

  const getStatusColor = (isActive: boolean) => {
    return isActive ? 'success' : 'error';
  };

  const handleDelete = () => {
    setDeleteDialogOpen(false);
    onDelete(model._id);
  };

  return (
    <>
      <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
        <CardContent sx={{ flexGrow: 1 }}>
          <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={2}>
            <Box>
              <Typography variant="h6" component="h2" gutterBottom>
                {model.name}
              </Typography>
              <Chip
                label={model.is_active ? 'Active' : 'Inactive'}
                color={getStatusColor(model.is_active)}
                size="small"
                sx={{ mb: 1 }}
              />
            </Box>
            <Box>
              <Tooltip title="Generate Text">
                <IconButton
                  size="small"
                  onClick={() => navigate(`/app/llm/generate/${model._id}`)}
                  sx={{ mr: 1 }}
                >
                  <PlayIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="View Metrics">
                <IconButton
                  size="small"
                  onClick={() => navigate(`/app/llm/metrics/${model._id}`)}
                  sx={{ mr: 1 }}
                >
                  <AssessmentIcon />
                </IconButton>
              </Tooltip>
              <IconButton
                onClick={() => setDeleteDialogOpen(true)}
                size="small"
                color="error"
              >
                <DeleteIcon />
              </IconButton>
            </Box>
          </Box>

          <Typography variant="body2" color="text.secondary" paragraph>
            {model.description || 'No description available'}
          </Typography>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            Type: {model.model_type}
          </Typography>
          
          <Typography variant="body2" color="text.secondary" gutterBottom>
            Version: {model.version}
          </Typography>

          {model.metadata && Object.keys(model.metadata).length > 0 && (
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>
                Metadata
              </Typography>
              <Grid container spacing={1}>
                {Object.entries(model.metadata).map(([key, value]) => (
                  <Grid item xs={6} key={key}>
                    <Typography variant="caption" color="textSecondary">
                      {key.replace('_', ' ').toUpperCase()}
                    </Typography>
                    <Typography variant="body2">
                      {typeof value === 'number' ? value.toFixed(4) : String(value)}
                    </Typography>
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
            Created {formatDistanceToNow(new Date(model.created_at))} ago
            {model.updated_at && ` • Updated ${formatDistanceToNow(new Date(model.updated_at))} ago`}
          </Typography>
        </CardContent>
      </Card>

      <Dialog open={deleteDialogOpen} onClose={() => setDeleteDialogOpen(false)}>
        <DialogTitle>Delete Model</DialogTitle>
        <DialogContent>
          <Typography>
            Are you sure you want to delete the model "{model.name}"? This action cannot be undone.
          </Typography>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDeleteDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleDelete} color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
    </>
  );
};
