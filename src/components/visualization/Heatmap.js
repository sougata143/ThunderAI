import React, { useState } from 'react';
import { Paper, Typography, FormControl, Select, MenuItem } from '@mui/material';
import { ResponsiveHeatMap } from '@nivo/heatmap';
import { useTheme } from '@mui/material/styles';

function Heatmap({ data, title, options = {} }) {
  const theme = useTheme();
  const [metric, setMetric] = useState(options.defaultMetric || 'accuracy');

  return (
    <Paper elevation={3} sx={{ p: 2, height: '400px' }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      {options.metrics && (
        <FormControl size="small" sx={{ mb: 2 }}>
          <Select
            value={metric}
            onChange={(e) => setMetric(e.target.value)}
          >
            {options.metrics.map(m => (
              <MenuItem key={m.value} value={m.value}>
                {m.label}
              </MenuItem>
            ))}
          </Select>
        </FormControl>
      )}
      <div style={{ height: '320px' }}>
        <ResponsiveHeatMap
          data={data}
          margin={{ top: 20, right: 90, bottom: 60, left: 90 }}
          valueFormat=">-.2f"
          axisTop={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: -45,
            legend: options.xAxisLabel || '',
            legendPosition: 'middle',
            legendOffset: -40
          }}
          axisRight={null}
          axisBottom={null}
          axisLeft={{
            tickSize: 5,
            tickPadding: 5,
            tickRotation: 0,
            legend: options.yAxisLabel || '',
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
          legends={[
            {
              anchor: 'bottom',
              translateX: 0,
              translateY: 30,
              length: 400,
              thickness: 8,
              direction: 'row',
              tickPosition: 'after',
              tickSize: 3,
              tickSpacing: 4,
              tickOverlap: false,
              title: metric,
              titleAlign: 'start',
              titleOffset: 4
            }
          ]}
        />
      </div>
    </Paper>
  );
}

export default Heatmap; 