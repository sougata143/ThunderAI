import React, { useEffect, useState, useCallback, useMemo } from 'react';
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
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import RealtimeVisualizationService from '../../services/realtimeVisualizationService';

function RealtimeChart({ 
  title,
  dataType,
  metrics = [],
  colors = {},
  timeWindow = 300000, // 5 minutes in milliseconds
}) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const processData = useCallback((newData) => {
    const now = Date.now();
    const cutoff = now - timeWindow;
    
    // Filter data within time window
    const filteredData = newData.filter(point => point.timestamp > cutoff);
    
    setData(filteredData);
    setLoading(false);
  }, [timeWindow]);

  useEffect(() => {
    RealtimeVisualizationService.connect(dataType);
    const unsubscribe = RealtimeVisualizationService.subscribe(dataType, processData);

    return () => {
      unsubscribe();
      RealtimeVisualizationService.disconnect();
    };
  }, [dataType, processData]);

  const formatXAxis = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString();
  };

  const chartData = useMemo(() => {
    return data.map(point => ({
      ...point,
      timestamp: new Date(point.timestamp).getTime()
    }));
  }, [data]);

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" height={400}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            dataKey="timestamp"
            tickFormatter={formatXAxis}
            domain={['dataMin', 'dataMax']}
            type="number"
          />
          <YAxis />
          <Tooltip
            labelFormatter={(label) => new Date(label).toLocaleString()}
          />
          <Legend />
          {metrics.map(metric => (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={colors[metric] || '#8884d8'}
              dot={false}
              isAnimationActive={false}
            />
          ))}
        </LineChart>
      </ResponsiveContainer>
    </Paper>
  );
}

export default RealtimeChart; 