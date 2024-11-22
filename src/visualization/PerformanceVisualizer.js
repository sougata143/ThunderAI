import React from 'react';
import {
  Grid,
  Paper,
  Typography
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar
} from 'recharts';

function PerformanceVisualizer({ metrics }) {
  return (
    <Grid container spacing={2}>
      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: 300 }}>
          <Typography variant="h6" gutterBottom>
            Training Progress
          </Typography>
          <ResponsiveContainer>
            <LineChart data={metrics.history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="accuracy" stroke="#82ca9d" />
              <Line type="monotone" dataKey="loss" stroke="#8884d8" />
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper sx={{ p: 2, height: 300 }}>
          <Typography variant="h6" gutterBottom>
            Model Performance
          </Typography>
          <ResponsiveContainer>
            <BarChart data={[metrics.final_metrics]}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="precision" fill="#8884d8" />
              <Bar dataKey="recall" fill="#82ca9d" />
              <Bar dataKey="f1_score" fill="#ffc658" />
            </BarChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default PerformanceVisualizer; 