import React, { useState, useEffect, useCallback } from 'react';
import { Paper, Typography, Box, CircularProgress } from '@mui/material';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import RealtimeVisualizationService from '../../services/realtimeVisualizationService';

function RealtimeHeatmap({
  title,
  dataType,
  xLabel,
  yLabel,
  aggregationWindow = 60000, // 1 minute window
}) {
  const [data, setData] = useState([]);
  const [loading, setLoading] = useState(true);

  const processData = useCallback((newData) => {
    // Aggregate data into heatmap format
    const aggregatedData = newData.reduce((acc, point) => {
      const xKey = point.x;
      const yKey = point.y;
      
      if (!acc[xKey]) {
        acc[xKey] = {};
      }
      if (!acc[xKey][yKey]) {
        acc[xKey][yKey] = 0;
      }
      acc[xKey][yKey] += point.value;
      return acc;
    }, {});

    // Transform into nivo heatmap format
    const formattedData = Object.entries(aggregatedData).map(([x, values]) => ({
      id: x,
      ...values
    }));

    setData(formattedData);
    setLoading(false);
  }, []);

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
      <Box height={400}>
        <ResponsiveHeatMap
          data={data}
          margin={{ top: 60, right: 90, bottom: 60, left: 90 }}
          valueFormat=">-.2f"
          axisTop={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: -45,
            legend: xLabel,
            legendPosition: 'middle',
            legendOffset: -40
          }}
          axisRight={null}
          axisBottom={null}
          axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: yLabel,
            legendPosition: 'middle',
            legendOffset: -60
          }}
          colors={{
            type: 'sequential',
            scheme: 'blues'
          }}
          emptyColor="#555555"
          borderColor={{ from: 'color', modifiers: [['darker', 0.6]] }}
          labelTextColor={{ from: 'color', modifiers: [['darker', 1.8]] }}
          animate={false}
        />
      </Box>
    </Paper>
  );
}

export default RealtimeHeatmap; 