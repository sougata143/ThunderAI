import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { instance as axios } from '../../api/axios';
import {
  Box,
  Typography,
  Grid,
  Button,
  CircularProgress,
  Card,
  CardContent,
  CardActions,
  Chip,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
} from '@mui/material';
import { Add as AddIcon } from '@mui/icons-material';

interface Experiment {
  id: string;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'pending';
  model: string;
  metrics: {
    accuracy?: number;
    loss?: number;
    [key: string]: number | undefined;
  };
  created_at: string;
  updated_at: string;
}

interface NewExperiment {
  name: string;
  description: string;
  model: string;
}

export const Experiments: React.FC = () => {
  const [experiments, setExperiments] = useState<Experiment[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [openNewDialog, setOpenNewDialog] = useState(false);
  const [newExperiment, setNewExperiment] = useState<NewExperiment>({
    name: '',
    description: '',
    model: '',
  });
  const navigate = useNavigate();
  const { user } = useAuth();

  useEffect(() => {
    if (user) {
      fetchExperiments();
    }
  }, [user]);

  const fetchExperiments = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/llm/experiments');
      setExperiments(response.data);
    } catch (err: any) {
      console.error('Full error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to fetch experiments');
    } finally {
      setLoading(false);
    }
  };

  const handleNewExperiment = async () => {
    try {
      const experimentData = {
        name: newExperiment.name,
        description: newExperiment.description,
        model: newExperiment.model,
        status: 'pending'
      };

      const response = await axios.post('/llm/experiments', experimentData);
      const createdExperiment = response.data;
      
      setExperiments([createdExperiment, ...experiments]);
      setOpenNewDialog(false);
      setNewExperiment({ name: '', description: '', model: '' });
      
      // Navigate to the new experiment's details page
      navigate(`/app/experiments/${createdExperiment.id}`);
    } catch (err: any) {
      console.error('Full error:', err);
      setError(err.response?.data?.detail || err.message || 'Failed to create experiment');
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'running':
        return 'primary';
      case 'completed':
        return 'success';
      case 'failed':
        return 'error';
      case 'pending':
        return 'warning';
      default:
        return 'default';
    }
  };

  if (loading) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', alignItems: 'center', minHeight: '60vh' }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Typography variant="h4" gutterBottom>
          Experiments
        </Typography>
        <Button
          variant="contained"
          color="primary"
          startIcon={<AddIcon />}
          onClick={() => setOpenNewDialog(true)}
        >
          New Experiment
        </Button>
      </Box>

      {error && (
        <Typography color="error" sx={{ mb: 2 }}>
          {error}
        </Typography>
      )}

      <Grid container spacing={3}>
        {experiments.map((experiment) => (
          <Grid item xs={12} md={6} lg={4} key={experiment.id}>
            <Card>
              <CardContent>
                <Typography variant="h6" gutterBottom>
                  {experiment.name}
                </Typography>
                <Typography variant="body2" color="textSecondary" gutterBottom>
                  {experiment.description}
                </Typography>
                <Box sx={{ mt: 2 }}>
                  <Chip
                    label={experiment.status}
                    color={getStatusColor(experiment.status)}
                    size="small"
                    sx={{ mr: 1 }}
                  />
                  <Chip label={experiment.model} size="small" />
                </Box>
                {experiment.metrics && (
                  <Box sx={{ mt: 2 }}>
                    {Object.entries(experiment.metrics).map(([key, value]) => (
                      <Typography key={key} variant="body2">
                        {key}: {value?.toFixed(4)}
                      </Typography>
                    ))}
                  </Box>
                )}
              </CardContent>
              <CardActions>
                <Button size="small" onClick={() => navigate(`/app/experiments/${experiment.id}`)}>
                  View Details
                </Button>
              </CardActions>
            </Card>
          </Grid>
        ))}
      </Grid>

      <Dialog open={openNewDialog} onClose={() => setOpenNewDialog(false)}>
        <DialogTitle>Create New Experiment</DialogTitle>
        <DialogContent>
          <TextField
            autoFocus
            margin="dense"
            label="Name"
            fullWidth
            value={newExperiment.name}
            onChange={(e) => setNewExperiment({ ...newExperiment, name: e.target.value })}
          />
          <TextField
            margin="dense"
            label="Description"
            fullWidth
            multiline
            rows={4}
            value={newExperiment.description}
            onChange={(e) => setNewExperiment({ ...newExperiment, description: e.target.value })}
          />
          <FormControl fullWidth margin="dense">
            <InputLabel>Model</InputLabel>
            <Select
              value={newExperiment.model}
              label="Model"
              onChange={(e) => setNewExperiment({ ...newExperiment, model: e.target.value })}
            >
              <MenuItem value="GPT-3">GPT-3</MenuItem>
              <MenuItem value="GPT-4">GPT-4</MenuItem>
              <MenuItem value="Custom">Custom</MenuItem>
            </Select>
          </FormControl>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setOpenNewDialog(false)}>Cancel</Button>
          <Button onClick={handleNewExperiment} variant="contained" color="primary">
            Create
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};
