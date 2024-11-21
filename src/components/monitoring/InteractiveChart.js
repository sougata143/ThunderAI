import React, { useState } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Button,
  ButtonGroup,
  Menu,
  MenuItem,
  IconButton,
  Tooltip
} from '@mui/material';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
  Legend, ResponsiveContainer, Brush, ZoomOutMap, ZoomIn, ZoomOut
} from 'recharts';
import DownloadIcon from '@mui/icons-material/Download';
import ShareIcon from '@mui/icons-material/Share';
import { CSVLink } from 'react-csv';

function InteractiveChart({ 
  data, 
  title, 
  metrics = [], 
  colors = {}, 
  timeRange = '1d',
  onTimeRangeChange 
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [zoomDomain, setZoomDomain] = useState(null);

  const handleExportClick = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleExportClose = () => {
    setAnchorEl(null);
  };

  const handleZoomIn = () => {
    if (!zoomDomain) {
      const length = data.length;
      setZoomDomain([length / 4, (length * 3) / 4]);
    } else {
      const range = zoomDomain[1] - zoomDomain[0];
      const center = (zoomDomain[0] + zoomDomain[1]) / 2;
      setZoomDomain([
        center - range / 4,
        center + range / 4
      ]);
    }
  };

  const handleZoomOut = () => {
    if (zoomDomain) {
      const range = zoomDomain[1] - zoomDomain[0];
      const center = (zoomDomain[0] + zoomDomain[1]) / 2;
      const newDomain = [
        Math.max(0, center - range),
        Math.min(data.length, center + range)
      ];
      setZoomDomain(newDomain);
    }
  };

  const handleResetZoom = () => {
    setZoomDomain(null);
  };

  const exportToCSV = () => {
    const csvData = data.map(item => ({
      timestamp: item.timestamp,
      ...metrics.reduce((acc, metric) => ({
        ...acc,
        [metric]: item[metric]
      }), {})
    }));
    return csvData;
  };

  const shareChart = async () => {
    try {
      const chartData = {
        title,
        data,
        metrics,
        timeRange
      };
      await navigator.clipboard.writeText(JSON.stringify(chartData));
      // Show success message using ErrorContext
    } catch (error) {
      // Show error message using ErrorContext
    }
  };

  return (
    <Paper elevation={3} sx={{ p: 2, mb: 2 }}>
      <Grid container spacing={2} alignItems="center">
        <Grid item xs>
          <Typography variant="h6">{title}</Typography>
        </Grid>
        <Grid item>
          <ButtonGroup size="small" sx={{ mr: 2 }}>
            <Button 
              onClick={() => onTimeRangeChange('1h')}
              variant={timeRange === '1h' ? 'contained' : 'outlined'}
            >
              1H
            </Button>
            <Button 
              onClick={() => onTimeRangeChange('1d')}
              variant={timeRange === '1d' ? 'contained' : 'outlined'}
            >
              1D
            </Button>
            <Button 
              onClick={() => onTimeRangeChange('1w')}
              variant={timeRange === '1w' ? 'contained' : 'outlined'}
            >
              1W
            </Button>
          </ButtonGroup>
          <ButtonGroup size="small" sx={{ mr: 2 }}>
            <Tooltip title="Zoom In">
              <IconButton onClick={handleZoomIn}>
                <ZoomIn />
              </IconButton>
            </Tooltip>
            <Tooltip title="Zoom Out">
              <IconButton onClick={handleZoomOut}>
                <ZoomOut />
              </IconButton>
            </Tooltip>
            <Tooltip title="Reset Zoom">
              <IconButton onClick={handleResetZoom}>
                <ZoomOutMap />
              </IconButton>
            </Tooltip>
          </ButtonGroup>
          <ButtonGroup size="small">
            <Tooltip title="Export">
              <IconButton onClick={handleExportClick}>
                <DownloadIcon />
              </IconButton>
            </Tooltip>
            <Tooltip title="Share">
              <IconButton onClick={shareChart}>
                <ShareIcon />
              </IconButton>
            </Tooltip>
          </ButtonGroup>
        </Grid>
      </Grid>

      <ResponsiveContainer width="100%" height={400}>
        <LineChart data={data}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis 
            dataKey="timestamp" 
            domain={zoomDomain ? ['dataMin', 'dataMax'] : undefined}
          />
          <YAxis />
          <RechartsTooltip />
          <Legend />
          {metrics.map(metric => (
            <Line
              key={metric}
              type="monotone"
              dataKey={metric}
              stroke={colors[metric] || '#8884d8'}
              dot={false}
            />
          ))}
          <Brush dataKey="timestamp" height={30} stroke="#8884d8" />
        </LineChart>
      </ResponsiveContainer>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleExportClose}
      >
        <CSVLink 
          data={exportToCSV()} 
          filename={`${title.toLowerCase().replace(/\s+/g, '-')}.csv`}
        >
          <MenuItem onClick={handleExportClose}>Export as CSV</MenuItem>
        </CSVLink>
        <MenuItem onClick={handleExportClose}>Export as PNG</MenuItem>
        <MenuItem onClick={handleExportClose}>Export as JSON</MenuItem>
      </Menu>
    </Paper>
  );
}

export default InteractiveChart; 