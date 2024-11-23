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

interface ModelCardProps {
  model: Model;
  onDelete: () => void;
}

export const ModelCard: React.FC<ModelCardProps> = ({ model, onDelete }) => {
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const navigate = useNavigate();

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'training':
        return 'warning';
      case 'ready':
        return 'success';
      case 'error':
        return 'error';
      default:
        return 'default';
    }
  };

  const formatMetric = (value: number) => {
    return value.toFixed(4);
  };

  const handleDelete = () => {
    setDeleteDialogOpen(false);
    onDelete();
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
                label={model.status}
                color={getStatusColor(model.status)}
                size="small"
                sx={{ mb: 1 }}
              />
            </Box>
            <Box>
              <Tooltip title="Generate Text">
                <IconButton
                  size="small"
                  onClick={() => navigate(`/llm/generate/${model.id}`)}
                  sx={{ mr: 1 }}
                >
                  <PlayIcon />
                </IconButton>
              </Tooltip>
              <Tooltip title="View Metrics">
                <IconButton
                  size="small"
                  onClick={() => navigate(`/llm/metrics/${model.id}`)}
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
            {model.description}
          </Typography>

          <Typography variant="body2" color="text.secondary" gutterBottom>
            Architecture: {model.architecture}
          </Typography>

          {model.metrics && (
            <Box mt={2}>
              <Typography variant="subtitle2" gutterBottom>
                Metrics
              </Typography>
              <Grid container spacing={1}>
                {Object.entries(model.metrics).map(([key, value]) => (
                  <Grid item xs={6} key={key}>
                    <Typography variant="caption" color="textSecondary">
                      {key.replace('_', ' ').toUpperCase()}
                    </Typography>
                    <Typography variant="body2">
                      {typeof value === 'number' ? formatMetric(value) : value}
                    </Typography>
                    {key === 'perplexity' && (
                      <LinearProgress
                        variant="determinate"
                        value={Math.min((1 / value) * 100, 100)}
                      />
                    )}
                    {key === 'accuracy' && (
                      <LinearProgress
                        variant="determinate"
                        value={value * 100}
                        color="success"
                      />
                    )}
                  </Grid>
                ))}
              </Grid>
            </Box>
          )}

          <Typography variant="caption" color="textSecondary" sx={{ mt: 2, display: 'block' }}>
            Created {formatDistanceToNow(new Date(model.created_at))} ago
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
