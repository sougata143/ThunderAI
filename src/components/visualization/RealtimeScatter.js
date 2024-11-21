import React, { useState, useEffect, useCallback } from 'react';
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import { ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import RealtimeVisualizationService from '../../services/realtimeVisualizationService';

function RealtimeScatter({
  title,
  dataType,
  xLabel,
  yLabel,
  categories = [],
  colors = {},
  windowSize = 100
}) {
  const [data, setData] = useState({});
  const [loading, setLoading] = useState(true);

  const processData = useCallback((newData) => {
    setData(prev => {
      const updated = { ...prev };
      
      // Group data by category
      newData.forEach(point => {
        const category = point.category;
        if (!updated[category]) {
          updated[category] = [];
        }
        
        updated[category].push({
          x: point.x,
          y: point.y,
          timestamp: point.timestamp
        });
        
        // Keep only the latest points
        if (updated[category].length > windowSize) {
          updated[category] = updated[category].slice(-windowSize);
        }
      });

      return updated;
    });
    
    setLoading(false);
  }, [windowSize]);

  useEffect(() => {
    RealtimeVisualizationService.connect(dataType);
    const unsubscribe = RealtimeVisualizationService.subscribe(dataType, processData);

    return () => {
      unsubscribe();
      RealtimeVisualizationService.disconnect();
    };
  }, [dataType, processData]);

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
        <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
          <CartesianGrid />
          <XAxis 
            type="number" 
            dataKey="x" 
            name={xLabel} 
            unit="" 
          />
          <YAxis 
            type="number" 
            dataKey="y" 
            name={yLabel} 
            unit="" 
          />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} />
          <Legend />
          {categories.map(category => (
            <Scatter
              key={category}
              name={category}
              data={data[category] || []}
              fill={colors[category] || '#8884d8'}
            />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
    </Paper>
  );
}

export default RealtimeScatter; 