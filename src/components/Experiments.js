import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  Button, 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableRow 
} from '@mui/material';
import axios from 'axios';

function Experiments() {
  const [experiments, setExperiments] = useState([]);
  const [activeExperiment, setActiveExperiment] = useState(null);

  useEffect(() => {
    fetchExperiments();
  }, []);

  const fetchExperiments = async () => {
    try {
      const response = await axios.get('/api/v1/experiments');
      setExperiments(response.data);
    } catch (error) {
      console.error('Error fetching experiments:', error);
    }
  };

  const startExperiment = async (modelA, modelB) => {
    try {
      const response = await axios.post('/api/v1/experiments', {
        model_a: modelA,
        model_b: modelB,
        traffic_split: 0.5
      });
      setActiveExperiment(response.data);
      fetchExperiments();
    } catch (error) {
      console.error('Error starting experiment:', error);
    }
  };

  const endExperiment = async (experimentId) => {
    try {
      await axios.post(`/api/v1/experiments/${experimentId}/end`);
      setActiveExperiment(null);
      fetchExperiments();
    } catch (error) {
      console.error('Error ending experiment:', error);
    }
  };

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">A/B Testing Experiments</Typography>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Active Experiments</Typography>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Experiment ID</TableCell>
                <TableCell>Model A</TableCell>
                <TableCell>Model B</TableCell>
                <TableCell>Status</TableCell>
                <TableCell>Actions</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {experiments.map((experiment) => (
                <TableRow key={experiment.id}>
                  <TableCell>{experiment.id}</TableCell>
                  <TableCell>{experiment.model_a}</TableCell>
                  <TableCell>{experiment.model_b}</TableCell>
                  <TableCell>{experiment.status}</TableCell>
                  <TableCell>
                    <Button 
                      variant="contained" 
                      color="secondary"
                      onClick={() => endExperiment(experiment.id)}
                    >
                      End Experiment
                    </Button>
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Experiments; 