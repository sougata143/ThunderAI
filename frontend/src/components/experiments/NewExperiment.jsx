import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Box,
  Card,
  CardContent,
  Typography,
  TextField,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Alert,
} from '@mui/material';
import ArrowBackIcon from '@mui/icons-material/ArrowBack';
import { createExperiment } from '../../api/experiments';

const NewExperiment = () => {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    model_type: '',
    status: 'pending',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);

    try {
      const experiment = await createExperiment(formData);
      navigate(`/experiments/${experiment.id}`);
    } catch (err) {
      setError('Failed to create experiment. Please try again.');
      console.error('Error creating experiment:', err);
    } finally {
      setLoading(false);
    }
  };

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

      <Card>
        <CardContent>
          <Typography variant="h5" component="h2" gutterBottom>
            Create New Experiment
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 3 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <TextField
                  required
                  fullWidth
                  label="Experiment Name"
                  name="name"
                  value={formData.name}
                  onChange={handleChange}
                  variant="outlined"
                />
              </Grid>

              <Grid item xs={12}>
                <TextField
                  fullWidth
                  label="Description"
                  name="description"
                  value={formData.description}
                  onChange={handleChange}
                  variant="outlined"
                  multiline
                  rows={4}
                />
              </Grid>

              <Grid item xs={12}>
                <FormControl fullWidth required>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    name="model_type"
                    value={formData.model_type}
                    onChange={handleChange}
                    label="Model Type"
                  >
                    <MenuItem value="classification">Classification</MenuItem>
                    <MenuItem value="regression">Regression</MenuItem>
                    <MenuItem value="clustering">Clustering</MenuItem>
                    <MenuItem value="nlp">Natural Language Processing</MenuItem>
                    <MenuItem value="computer_vision">Computer Vision</MenuItem>
                  </Select>
                </FormControl>
              </Grid>

              <Grid item xs={12}>
                <Box display="flex" justifyContent="flex-end" gap={2}>
                  <Button
                    variant="outlined"
                    onClick={() => navigate('/')}
                  >
                    Cancel
                  </Button>
                  <Button
                    type="submit"
                    variant="contained"
                    color="primary"
                    disabled={loading}
                  >
                    {loading ? 'Creating...' : 'Create Experiment'}
                  </Button>
                </Box>
              </Grid>
            </Grid>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
};

export { NewExperiment };
