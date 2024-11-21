import React, { useEffect, useState } from 'react';
import { Grid, Paper, Typography } from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  AreaChart, Area, BarChart, Bar
} from 'recharts';
import websocket from '../../services/websocket';

function RealTimeMetrics() {
  const [metrics, setMetrics] = useState({
    modelPerformance: [],
    systemResources: [],
    predictions: []
  });

  useEffect(() => {
    websocket.connect();

    websocket.subscribe('model_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        modelPerformance: [...prev.modelPerformance, data].slice(-20)
      }));
    });

    websocket.subscribe('system_metrics', (data) => {
      setMetrics(prev => ({
        ...prev,
        systemResources: [...prev.systemResources, data].slice(-20)
      }));
    });

    websocket.subscribe('predictions', (data) => {
      setMetrics(prev => ({
        ...prev,
        predictions: [...prev.predictions, data].slice(-20)
      }));
    });

    return () => {
      websocket.unsubscribe('model_metrics');
      websocket.unsubscribe('system_metrics');
      websocket.unsubscribe('predictions');
    };
  }, []);

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Real-Time Monitoring</Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Model Performance</Typography>
          <AreaChart width={500} height={300} data={metrics.modelPerformance}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Area 
              type="monotone" 
              dataKey="accuracy" 
              stroke="#8884d8" 
              fill="#8884d8" 
              fillOpacity={0.3} 
            />
            <Area 
              type="monotone" 
              dataKey="latency" 
              stroke="#82ca9d" 
              fill="#82ca9d" 
              fillOpacity={0.3} 
            />
          </AreaChart>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">System Resources</Typography>
          <LineChart width={500} height={300} data={metrics.systemResources}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="timestamp" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Line type="monotone" dataKey="cpu" stroke="#8884d8" />
            <Line type="monotone" dataKey="memory" stroke="#82ca9d" />
            <Line type="monotone" dataKey="gpu" stroke="#ffc658" />
          </LineChart>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Prediction Distribution</Typography>
          <BarChart width={1000} height={300} data={metrics.predictions}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="model" />
            <YAxis />
            <Tooltip />
            <Legend />
            <Bar dataKey="correct" stackId="a" fill="#82ca9d" />
            <Bar dataKey="incorrect" stackId="a" fill="#ff8042" />
          </BarChart>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default RealTimeMetrics; 