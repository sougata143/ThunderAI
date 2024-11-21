import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  Table, 
  TableBody, 
  TableCell, 
  TableHead, 
  TableRow,
  CircularProgress
} from '@mui/material';
import {
  RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar, Legend
} from 'recharts';
import axios from 'axios';

function ModelComparison() {
  const [modelMetrics, setModelMetrics] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const response = await axios.get('/api/v1/models/metrics/comparison');
        setModelMetrics(response.data);
      } catch (err) {
        setError('Failed to fetch model metrics');
        console.error('Error fetching metrics:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, []);

  if (loading) return <CircularProgress />;
  if (error) return <Typography color="error">{error}</Typography>;

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Model Comparison</Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Performance Metrics</Typography>
          <RadarChart width={500} height={400} data={modelMetrics}>
            <PolarGrid />
            <PolarAngleAxis dataKey="metric" />
            <PolarRadiusAxis angle={30} domain={[0, 1]} />
            {modelMetrics.map((model, index) => (
              <Radar
                key={model.name}
                name={model.name}
                dataKey="value"
                stroke={`#${Math.floor(Math.random()*16777215).toString(16)}`}
                fill={`#${Math.floor(Math.random()*16777215).toString(16)}`}
                fillOpacity={0.6}
              />
            ))}
            <Legend />
          </RadarChart>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Detailed Metrics</Typography>
          <Table>
            <TableHead>
              <TableRow>
                <TableCell>Model</TableCell>
                <TableCell>Accuracy</TableCell>
                <TableCell>Latency</TableCell>
                <TableCell>Memory Usage</TableCell>
              </TableRow>
            </TableHead>
            <TableBody>
              {modelMetrics.map((model) => (
                <TableRow key={model.name}>
                  <TableCell>{model.name}</TableCell>
                  <TableCell>{model.accuracy.toFixed(3)}</TableCell>
                  <TableCell>{model.latency.toFixed(3)}ms</TableCell>
                  <TableCell>{model.memoryUsage}MB</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelComparison; 