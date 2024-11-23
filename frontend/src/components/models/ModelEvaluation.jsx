import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  Grid,
  Box,
  Tabs,
  Tab,
  CircularProgress,
  Card,
  CardContent,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
} from '@mui/material';

// Mock data - replace with actual API calls
const mockMetrics = {
  accuracy: 0.92,
  precision: 0.89,
  recall: 0.94,
  f1Score: 0.91,
  auc: 0.95,
};

const mockConfusionMatrix = [
  [150, 10],
  [15, 125],
];

const mockFeatureImportance = [
  { feature: 'feature_1', importance: 0.35 },
  { feature: 'feature_2', importance: 0.25 },
  { feature: 'feature_3', importance: 0.20 },
  { feature: 'feature_4', importance: 0.15 },
  { feature: 'feature_5', importance: 0.05 },
];

const TabPanel = ({ children, value, index }) => (
  <div hidden={value !== index} role="tabpanel">
    {value === index && <Box sx={{ p: 3 }}>{children}</Box>}
  </div>
);

const ModelEvaluation = () => {
  const { modelId } = useParams();
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState(0);
  const [metrics, setMetrics] = useState(null);
  const [confusionMatrix, setConfusionMatrix] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);

  useEffect(() => {
    const fetchEvaluationData = async () => {
      try {
        // Replace with actual API calls
        setMetrics(mockMetrics);
        setConfusionMatrix(mockConfusionMatrix);
        setFeatureImportance(mockFeatureImportance);
      } catch (error) {
        console.error('Failed to fetch evaluation data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchEvaluationData();
  }, [modelId]);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="80vh">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Model Evaluation
        </Typography>

        <Box sx={{ borderBottom: 1, borderColor: 'divider', mb: 3 }}>
          <Tabs value={activeTab} onChange={handleTabChange}>
            <Tab label="Overview" />
            <Tab label="Metrics" />
            <Tab label="Confusion Matrix" />
            <Tab label="Feature Importance" />
          </Tabs>
        </Box>

        {/* Overview Tab */}
        <TabPanel value={activeTab} index={0}>
          <Grid container spacing={3}>
            {Object.entries(metrics).map(([key, value]) => (
              <Grid item xs={12} sm={6} md={4} key={key}>
                <Card>
                  <CardContent>
                    <Typography color="textSecondary" gutterBottom>
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </Typography>
                    <Typography variant="h4">
                      {typeof value === 'number' ? value.toFixed(2) : value}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </TabPanel>

        {/* Detailed Metrics Tab */}
        <TabPanel value={activeTab} index={1}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Metric</TableCell>
                  <TableCell align="right">Value</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {Object.entries(metrics).map(([key, value]) => (
                  <TableRow key={key}>
                    <TableCell component="th" scope="row">
                      {key.charAt(0).toUpperCase() + key.slice(1)}
                    </TableCell>
                    <TableCell align="right">
                      {typeof value === 'number' ? value.toFixed(4) : value}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Confusion Matrix Tab */}
        <TabPanel value={activeTab} index={2}>
          <TableContainer component={Paper}>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell />
                  <TableCell align="center">Predicted Negative</TableCell>
                  <TableCell align="center">Predicted Positive</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                <TableRow>
                  <TableCell component="th" scope="row">
                    Actual Negative
                  </TableCell>
                  {confusionMatrix[0].map((value, index) => (
                    <TableCell key={index} align="center">
                      {value}
                    </TableCell>
                  ))}
                </TableRow>
                <TableRow>
                  <TableCell component="th" scope="row">
                    Actual Positive
                  </TableCell>
                  {confusionMatrix[1].map((value, index) => (
                    <TableCell key={index} align="center">
                      {value}
                    </TableCell>
                  ))}
                </TableRow>
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>

        {/* Feature Importance Tab */}
        <TabPanel value={activeTab} index={3}>
          <TableContainer>
            <Table>
              <TableHead>
                <TableRow>
                  <TableCell>Feature</TableCell>
                  <TableCell align="right">Importance</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {featureImportance.map((feature) => (
                  <TableRow key={feature.feature}>
                    <TableCell component="th" scope="row">
                      {feature.feature}
                    </TableCell>
                    <TableCell align="right">
                      {feature.importance.toFixed(4)}
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </TabPanel>
      </Paper>
    </Container>
  );
};

export default ModelEvaluation;
