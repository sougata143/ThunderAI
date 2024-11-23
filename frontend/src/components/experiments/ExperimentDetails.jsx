import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Button,
  Chip,
  CircularProgress,
  Grid,
  Paper,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { getExperiment } from '../../api/experiments';

const getStatusColor = (status) => {
  switch (status.toLowerCase()) {
    case 'completed':
      return 'success';
    case 'in_progress':
      return 'warning';
    case 'failed':
      return 'error';
    case 'pending':
      return 'info';
    default:
      return 'default';
  }
};

const formatDate = (dateString) => {
  return new Date(dateString).toLocaleString();
};

const ExperimentDetails = () => {
  const { id } = useParams();
  const navigate = useNavigate();
  const [experiment, setExperiment] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadExperiment = async () => {
      try {
        const data = await getExperiment(parseInt(id));
        setExperiment(data);
        setError(null);
      } catch (err) {
        setError('Failed to load experiment details');
        console.error('Error loading experiment:', err);
      } finally {
        setLoading(false);
      }
    };

    loadExperiment();
  }, [id]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  if (error) {
    return (
      <Box p={3}>
        <Typography color="error">{error}</Typography>
        <Button
          variant="contained"
          onClick={() => navigate('/')}
          startIcon={<ArrowBackIcon />}
          sx={{ mt: 2 }}
        >
          Back to Experiments
        </Button>
      </Box>
    );
  }

  if (!experiment) {
    return (
      <Box p={3}>
        <Typography>Experiment not found</Typography>
        <Button
          variant="contained"
          onClick={() => navigate('/')}
          startIcon={<ArrowBackIcon />}
          sx={{ mt: 2 }}
        >
          Back to Experiments
        </Button>
      </Box>
    );
  }

  return (
    <Box p={3}>
      <Button
        variant="outlined"
        onClick={() => navigate('/')}
        startIcon={<ArrowBackIcon />}
        sx={{ mb: 3 }}
      >
        Back to Experiments
      </Button>

      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Card>
            <CardContent>
              <Box display="flex" justifyContent="space-between" alignItems="center" mb={3}>
                <Typography variant="h5" component="h2">
                  {experiment.name}
                </Typography>
                <Chip
                  label={experiment.status}
                  color={getStatusColor(experiment.status)}
                  variant="outlined"
                />
              </Box>

              <Grid container spacing={3}>
                <Grid item xs={12}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Description
                    </Typography>
                    <Typography variant="body1">
                      {experiment.description || 'No description provided'}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Model Type
                    </Typography>
                    <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                      {experiment.model_type.replace('_', ' ')}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Status
                    </Typography>
                    <Typography variant="body1" sx={{ textTransform: 'capitalize' }}>
                      {experiment.status.replace('_', ' ')}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Created At
                    </Typography>
                    <Typography variant="body1">
                      {formatDate(experiment.created_at)}
                    </Typography>
                  </Paper>
                </Grid>

                <Grid item xs={12} md={6}>
                  <Paper elevation={0} sx={{ p: 2, bgcolor: 'grey.50' }}>
                    <Typography variant="subtitle2" color="text.secondary" gutterBottom>
                      Last Updated
                    </Typography>
                    <Typography variant="body1">
                      {formatDate(experiment.updated_at)}
                    </Typography>
                  </Paper>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        </Grid>
      </Grid>
    </Box>
  );
};

export { ExperimentDetails };
