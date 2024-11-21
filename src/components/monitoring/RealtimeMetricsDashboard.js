import React from 'react';
import { Grid } from '@mui/material';
import RealtimeChart from '../visualization/RealtimeChart';

const METRICS_CONFIG = {
  performance: {
    title: 'Model Performance',
    metrics: ['accuracy', 'loss'],
    colors: {
      accuracy: '#82ca9d',
      loss: '#8884d8'
    }
  },
  resources: {
    title: 'System Resources',
    metrics: ['cpu', 'memory', 'gpu'],
    colors: {
      cpu: '#8884d8',
      memory: '#82ca9d',
      gpu: '#ffc658'
    }
  },
  predictions: {
    title: 'Predictions',
    metrics: ['total', 'successful', 'failed'],
    colors: {
      total: '#8884d8',
      successful: '#82ca9d',
      failed: '#ff8042'
    }
  },
  latency: {
    title: 'Response Latency',
    metrics: ['p50', 'p90', 'p99'],
    colors: {
      p50: '#8884d8',
      p90: '#82ca9d',
      p99: '#ff8042'
    }
  }
};

function RealtimeMetricsDashboard() {
  return (
    <Grid container spacing={3} padding={3}>
      {Object.entries(METRICS_CONFIG).map(([type, config]) => (
        <Grid item xs={12} md={6} key={type}>
          <RealtimeChart
            title={config.title}
            dataType={type}
            metrics={config.metrics}
            colors={config.colors}
          />
        </Grid>
      ))}
    </Grid>
  );
}

export default RealtimeMetricsDashboard; 