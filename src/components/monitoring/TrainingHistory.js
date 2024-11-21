import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import axios from 'axios';
import LoadingSkeleton from '../common/LoadingSkeleton';

function TrainingHistory() {
  const [history, setHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedModel, setSelectedModel] = useState('all');
  const [selectedMetric, setSelectedMetric] = useState('loss');

  useEffect(() => {
    fetchTrainingHistory();
  }, [selectedModel]);

  const fetchTrainingHistory = async () => {
    try {
      setLoading(true);
      const response = await axios.get('/api/v1/models/training-history', {
        params: {
          model: selectedModel !== 'all' ? selectedModel : undefined
        }
      });
      setHistory(response.data);
      setError(null);
    } catch (err) {
      setError('Failed to fetch training history');
      console.error('Error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) return <LoadingSkeleton type="combined" />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Grid container spacing={3} padding={3}>
      <Grid item xs={12}>
        <Typography variant="h4">Model Training History</Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Model</InputLabel>
          <Select
            value={selectedModel}
            onChange={(e) => setSelectedModel(e.target.value)}
          >
            <MenuItem value="all">All Models</MenuItem>
            <MenuItem value="bert">BERT</MenuItem>
            <MenuItem value="gpt">GPT</MenuItem>
            <MenuItem value="lstm">LSTM</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12} md={6}>
        <FormControl fullWidth>
          <InputLabel>Metric</InputLabel>
          <Select
            value={selectedMetric}
            onChange={(e) => setSelectedMetric(e.target.value)}
          >
            <MenuItem value="loss">Loss</MenuItem>
            <MenuItem value="accuracy">Accuracy</MenuItem>
            <MenuItem value="val_loss">Validation Loss</MenuItem>
            <MenuItem value="val_accuracy">Validation Accuracy</MenuItem>
          </Select>
        </FormControl>
      </Grid>

      <Grid item xs={12}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6">Training Progress</Typography>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="epoch" />
              <YAxis />
              <Tooltip />
              <Legend />
              {history.map((model, index) => (
                <Line
                  key={model.name}
                  type="monotone"
                  dataKey={selectedMetric}
                  name={model.name}
                  stroke={`#${Math.floor(Math.random()*16777215).toString(16)}`}
                  dot={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default TrainingHistory; 