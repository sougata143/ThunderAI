import React, { useRef, useEffect, useState } from 'react';
import { Paper, Typography, FormControl, Select, MenuItem, Slider } from '@mui/material';
import { Network } from 'vis-network';
import { useTheme } from '@mui/material/styles';

function NetworkGraph({ data, title, options = {} }) {
  const containerRef = useRef(null);
  const [network, setNetwork] = useState(null);
  const [zoomLevel, setZoomLevel] = useState(1);
  const theme = useTheme();

  useEffect(() => {
    if (containerRef.current && data) {
      const networkOptions = {
        nodes: {
          shape: 'dot',
          size: 16,
          font: {
            size: 12,
            color: theme.palette.text.primary
          },
          borderWidth: 2
        },
        edges: {
          width: 1,
          color: { inherit: 'from' },
          smooth: {
            type: 'continuous'
          }
        },
        physics: {
          stabilization: false,
          barnesHut: {
            gravitationalConstant: -80000,
            springConstant: 0.001,
            springLength: 200
          }
        },
        interaction: {
          navigationButtons: true,
          keyboard: true
        },
        ...options
      };

      const newNetwork = new Network(containerRef.current, data, networkOptions);
      setNetwork(newNetwork);

      newNetwork.on('zoom', (params) => {
        setZoomLevel(params.scale);
      });

      return () => {
        newNetwork.destroy();
      };
    }
  }, [data, options, theme]);

  const handleZoomChange = (_, value) => {
    if (network) {
      network.moveTo({
        scale: value,
        animation: true
      });
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <div style={{ display: 'flex', alignItems: 'center', mb: 2 }}>
        <Typography variant="body2" sx={{ mr: 2 }}>
          Zoom:
        </Typography>
        <Slider
          value={zoomLevel}
          onChange={handleZoomChange}
          min={0.1}
          max={2}
          step={0.1}
          sx={{ width: 200 }}
        />
      </div>
      <div ref={containerRef} style={{ height: '400px' }} />
    </Paper>
  );
}

export default NetworkGraph; 