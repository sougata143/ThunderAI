import React, { useRef, useEffect, useState } from 'react';
import { Paper, Typography, FormControl, Select, MenuItem } from '@mui/material';
import { Graph3D } from 'vis-graph3d';

function ScatterPlot3D({ data, title, options = {} }) {
  const containerRef = useRef(null);
  const [graph, setGraph] = useState(null);
  const [selectedMetrics, setSelectedMetrics] = useState({
    x: options.defaultMetrics?.x || 'accuracy',
    y: options.defaultMetrics?.y || 'latency',
    z: options.defaultMetrics?.z || 'memory'
  });

  useEffect(() => {
    if (containerRef.current && !graph) {
      const graphOptions = {
        width: '100%',
        height: '400px',
        style: 'dot-color',
        showPerspective: true,
        showGrid: true,
        showShadow: false,
        keepAspectRatio: true,
        verticalRatio: 0.5,
        xLabel: selectedMetrics.x,
        yLabel: selectedMetrics.y,
        zLabel: selectedMetrics.z,
        ...options
      };

      const newGraph = new Graph3D(containerRef.current, data, graphOptions);
      setGraph(newGraph);
    }
  }, [containerRef, data, options]);

  useEffect(() => {
    if (graph) {
      const formattedData = data.map(point => ({
        x: point[selectedMetrics.x],
        y: point[selectedMetrics.y],
        z: point[selectedMetrics.z],
        style: point.style || 0
      }));
      graph.setData(formattedData);
    }
  }, [selectedMetrics, data, graph]);

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <div style={{ display: 'flex', gap: 2, mb: 2 }}>
        {['x', 'y', 'z'].map(axis => (
          <FormControl key={axis} size="small">
            <Select
              value={selectedMetrics[axis]}
              onChange={(e) => setSelectedMetrics(prev => ({
                ...prev,
                [axis]: e.target.value
              }))}
            >
              {options.metrics?.map(m => (
                <MenuItem key={m.value} value={m.value}>
                  {`${axis.toUpperCase()}: ${m.label}`}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        ))}
      </div>
      <div ref={containerRef} style={{ height: '400px' }} />
    </Paper>
  );
}

export default ScatterPlot3D; 