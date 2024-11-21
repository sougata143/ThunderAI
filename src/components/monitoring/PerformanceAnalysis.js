import React, { useState, useEffect } from 'react';
import { 
  Grid, 
  Paper, 
  Typography, 
  CircularProgress, 
  Alert,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend,
  ResponsiveContainer, ZAxis, Label
} from 'recharts';
import axios from 'axios';

function PerformanceAnalysis() {
  const [performanceData, setPerformanceData] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedMetric, setSelectedMetric] = useState('accuracy');
  const [timeRange, setTimeRange] = useState('24h');

  useEffect(() => {
    fetchPerformanceData();
  }, [selectedMetric, timeRange]);

  const fetchPerformanceData = async () => {
    try {
      setLoading(true);
      const response = await axios.get(`/api/v1/metrics/performance`, {
        params: {
          metric: selectedMetric,
          timeRange: timeRange
        }
      });
      setPerformanceData(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch performance data');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <Grid container justifyContent="center" alignItems="center" style={{ minHeight: '400px' }}>
        <CircularProgress />
      </Grid>
    );
  }

  if (error) {
    return (
      <Alert severity="error" sx={{ mt: 2 }}>
        {error}
      </Alert>
    );
  }

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Performance Analysis</Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Metric</InputLabel>
          <Select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            <MenuItem value="accuracy">Accuracy</MenuItem>
            <MenuItem value="latency">Latency</MenuItem>
            <MenuItem value="memory">Memory Usage</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Time Range</InputLabel>
          <Select
            value={timeRange}
            onChange={(e) => setTimeRange(e.target.value)}
          >
            <MenuItem value="24h">Last 24 Hours</MenuItem>
            <MenuItem value="7d">Last 7 Days</MenuItem>
            <MenuItem value="30d">Last 30 Days</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Performance Distribution</Typography>
          <ResponsiveContainer width="100%" height={400}>
            <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
              <CartesianGrid />
              <XAxis 
                dataKey="timestamp" 
                type="number" 
                domain={['auto', 'auto']}
                name="Time"
              >
                <Label value="Time" offset={0} position="bottom" />
              </XAxis>
              <YAxis 
                dataKey={selectedMetric} 
                name={selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)}
              >
                <Label 
                  value={selectedMetric.charAt(0).toUpperCase() + selectedMetric.slice(1)} 
                  angle={-90} 
                  position="left" 
                />
              </YAxis>
              <ZAxis dataKey="size" range={[64, 144]} />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} />
              <Legend />
              {performanceData.map((model, index) => (
                <Scatter
                  key={model.name}
                  name={model.name}
                  data={model.data}
                  fill={`#${Math.floor(Math.random()*16777215).toString(16)}`}
                />
              ))}
            </ScatterChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default PerformanceAnalysis; 