import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  Alert,
  CircularProgress,
  Card,
  CardContent,
  Tabs,
  Tab,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Chip
} from '@mui/material';
import { useSelector } from 'react-redux';
import { modelService } from '../../services/modelService';
import { InteractiveChart } from '../../visualization';

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
    </div>
  );
}

function ModelMonitoring() {
  const [tabValue, setTabValue] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [predictions, setPredictions] = useState([]);
  const [resourceMetrics, setResourceMetrics] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const modelId = useSelector(state => state.training.modelId);

  useEffect(() => {
    if (modelId) {
      fetchMetrics();
      const interval = setInterval(fetchMetrics, 30000); // Refresh every 30 seconds
      return () => clearInterval(interval);
    }
  }, [modelId]);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const [metricsResponse, predictionsResponse, resourceResponse] = await Promise.all([
        modelService.getModelMetrics(modelId),
        modelService.getRecentPredictions(modelId),
        modelService.getResourceMetrics(modelId)
      ]);
      
      setMetrics(metricsResponse.data);
      setPredictions(predictionsResponse.data);
      setResourceMetrics(resourceResponse.data);
    } catch (err) {
      setError('Failed to fetch monitoring data');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleTabChange = (event, newValue) => {
    setTabValue(newValue);
  };

  const renderPerformanceMetrics = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Accuracy Over Time
            </Typography>
            {metrics && (
              <InteractiveChart
                data={metrics.accuracy_history}
                title="Model Accuracy"
                xAxis="timestamp"
                series={['accuracy']}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Latency Distribution
            </Typography>
            {metrics && (
              <InteractiveChart
                data={metrics.latency_distribution}
                title="Response Time (ms)"
                xAxis="timestamp"
                series={['p50', 'p95', 'p99']}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderResourceUsage = () => (
    <Grid container spacing={3}>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              CPU Usage
            </Typography>
            {resourceMetrics && (
              <InteractiveChart
                data={resourceMetrics.cpu_usage}
                title="CPU Utilization (%)"
                xAxis="timestamp"
                series={['usage']}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12} md={6}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Memory Usage
            </Typography>
            {resourceMetrics && (
              <InteractiveChart
                data={resourceMetrics.memory_usage}
                title="Memory Usage (MB)"
                xAxis="timestamp"
                series={['usage']}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
      <Grid item xs={12}>
        <Card>
          <CardContent>
            <Typography variant="h6" gutterBottom>
              Network Traffic
            </Typography>
            {resourceMetrics && (
              <InteractiveChart
                data={resourceMetrics.network_traffic}
                title="Network Traffic"
                xAxis="timestamp"
                series={['incoming', 'outgoing']}
              />
            )}
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );

  const renderPredictions = () => (
    <TableContainer component={Paper}>
      <Table>
        <TableHead>
          <TableRow>
            <TableCell>Timestamp</TableCell>
            <TableCell>Input</TableCell>
            <TableCell>Prediction</TableCell>
            <TableCell>Confidence</TableCell>
            <TableCell>Latency (ms)</TableCell>
            <TableCell>Status</TableCell>
          </TableRow>
        </TableHead>
        <TableBody>
          {predictions.map((pred) => (
            <TableRow key={pred.id}>
              <TableCell>{new Date(pred.timestamp).toLocaleString()}</TableCell>
              <TableCell>{pred.input.substring(0, 50)}...</TableCell>
              <TableCell>{pred.prediction}</TableCell>
              <TableCell>{(pred.confidence * 100).toFixed(2)}%</TableCell>
              <TableCell>{pred.latency}</TableCell>
              <TableCell>
                <Chip
                  label={pred.status}
                  color={pred.status === 'success' ? 'success' : 'error'}
                  size="small"
                />
              </TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </TableContainer>
  );

  if (!modelId) {
    return (
      <Alert severity="info">
        No model selected. Please select a model to monitor.
      </Alert>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Model Monitoring Dashboard
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 2 }}>
            <Tabs value={tabValue} onChange={handleTabChange}>
              <Tab label="Performance Metrics" />
              <Tab label="Resource Usage" />
              <Tab label="Predictions" />
            </Tabs>
          </Box>

          {loading ? (
            <Box display="flex" justifyContent="center" p={3}>
              <CircularProgress />
            </Box>
          ) : (
            <>
              <TabPanel value={tabValue} index={0}>
                {renderPerformanceMetrics()}
              </TabPanel>
              <TabPanel value={tabValue} index={1}>
                {renderResourceUsage()}
              </TabPanel>
              <TabPanel value={tabValue} index={2}>
                {renderPredictions()}
              </TabPanel>
            </>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelMonitoring; 