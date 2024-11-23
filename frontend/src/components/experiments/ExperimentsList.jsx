import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Grid,
  Chip,
  CircularProgress
} from '@mui/material';
import { fetchExperiments } from '../../api/experiments';

const getStatusColor = (status) => {
  switch (status.toLowerCase()) {
    case 'completed':
      return 'success';
    case 'in_progress':
      return 'warning';
    case 'failed':
      return 'error';
    default:
      return 'default';
  }
};

const ExperimentCard = ({ experiment, onViewDetails }) => (
  <Card>
    <CardContent>
      <Typography variant="h6" component="h2" gutterBottom>
        {experiment.name}
      </Typography>
      <Box display="flex" justifyContent="space-between" alignItems="center">
        <Typography color="textSecondary">
          ID: {experiment.id}
        </Typography>
        <Chip
          label={experiment.status}
          color={getStatusColor(experiment.status)}
          variant="outlined"
          size="small"
        />
      </Box>
    </CardContent>
    <CardActions>
      <Button 
        size="small" 
        color="primary" 
        onClick={() => onViewDetails(experiment.id)}
      >
        View Details
      </Button>
    </CardActions>
  </Card>
);

const ExperimentsList = () => {
  const navigate = useNavigate();
  const [experiments, setExperiments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadExperiments = async () => {
      try {
        const data = await fetchExperiments();
        setExperiments(data);
        setError(null);
      } catch (err) {
        setError('Failed to load experiments');
        console.error('Error loading experiments:', err);
      } finally {
        setLoading(false);
      }
    };

    loadExperiments();
  }, []);

  const handleViewDetails = (experimentId) => {
    navigate(`/experiments/${experimentId}`);
  };

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
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      {experiments.map((experiment) => (
        <Grid item xs={12} sm={6} md={4} key={experiment.id}>
          <ExperimentCard 
            experiment={experiment} 
            onViewDetails={handleViewDetails}
          />
        </Grid>
      ))}
    </Grid>
  );
};

export default ExperimentsList;
