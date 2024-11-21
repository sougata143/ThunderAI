import React, { useState, useEffect } from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend } from 'recharts';
import axios from 'axios';

function Dashboard() {
  const [metrics, setMetrics] = useState({
    modelAccuracy: [],
    predictionLatency: [],
    systemMetrics: []
  });

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [accuracyRes, latencyRes, systemRes] = await Promise.all([
          axios.get('/api/v1/metrics/accuracy'),
          axios.get('/api/v1/metrics/latency'),
          axios.get('/api/v1/metrics/system')
        ]);

        setMetrics({
          modelAccuracy: accuracyRes.data,
          predictionLatency: latencyRes.data,
          systemMetrics: systemRes.data
        });
      } catch (error) {
        console.error('Error fetching metrics:', error);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">System Dashboard</Typography>
      </Grid>
      
      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Model Accuracy</Typography>
          <LineChart width={500} height={300} data={metrics.modelAccuracy}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="accuracy" stroke="#8884d8" />
          </LineChart>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Prediction Latency</Typography>
          <LineChart width={500} height={300} data={metrics.predictionLatency}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="latency" stroke="#82ca9d" />
          </LineChart>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Dashboard; 