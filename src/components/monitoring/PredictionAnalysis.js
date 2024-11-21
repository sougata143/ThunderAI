import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Alert,
  Box,
  Tabs,
  Tab
} from '@mui/material';
import {
  ComposedChart,
  Line,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell
} from 'recharts';
import axios from 'axios';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042'];

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function PredictionAnalysis() {
  const [value, setValue] = useState(0);
  const [predictionData, setPredictionData] = useState({
    timeline: [],
    distribution: [],
    accuracy: []
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchPredictionData();
  }, []);

  const fetchPredictionData = async () => {
    try {
      setLoading(true);
      const [timelineRes, distributionRes, accuracyRes] = await Promise.all([
        axios.get('/api/v1/metrics/predictions/timeline'),
        axios.get('/api/v1/metrics/predictions/distribution'),
        axios.get('/api/v1/metrics/predictions/accuracy')
      ]);

      setPredictionData({
        timeline: timelineRes.data,
        distribution: distributionRes.data,
        accuracy: accuracyRes.data
      });
      setError(null);
    } catch (err) {
      setError('Failed to fetch prediction data');
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
        <Typography variant="h4">Prediction Analysis</Typography>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3}>
          <Tabs
            value={value}
            onChange={(e, newValue) => setValue(newValue)}
            centered
          >
            <Tab label="Timeline" />
            <Tab label="Distribution" />
            <Tab label="Accuracy" />
          </Tabs>

          <TabPanel value={value} index={0}>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={predictionData.timeline}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="timestamp" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="count" fill="#8884d8" />
                <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#82ca9d" />
              </ComposedChart>
            </ResponsiveContainer>
          </TabPanel>

          <TabPanel value={value} index={1}>
            <ResponsiveContainer width="100%" height={400}>
              <PieChart>
                <Pie
                  data={predictionData.distribution}
                  dataKey="value"
                  nameKey="name"
                  cx="50%"
                  cy="50%"
                  outerRadius={150}
                  label
                >
                  {predictionData.distribution.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </TabPanel>

          <TabPanel value={value} index={2}>
            <ResponsiveContainer width="100%" height={400}>
              <ComposedChart data={predictionData.accuracy}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="model" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="accuracy" fill="#8884d8" />
                <Line type="monotone" dataKey="trend" stroke="#82ca9d" />
              </ComposedChart>
            </ResponsiveContainer>
          </TabPanel>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default PredictionAnalysis; 