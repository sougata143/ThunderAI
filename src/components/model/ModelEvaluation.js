import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Card,
  CardContent
} from '@mui/material';
import { useSelector } from 'react-redux';
import { modelService } from '../../services/modelService';
import { InteractiveChart } from '../../visualization';

function ModelEvaluation() {
  const [evaluating, setEvaluating] = useState(false);
  const [error, setError] = useState(null);
  const [results, setResults] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const modelId = useSelector(state => state.training.modelId);
  const trainingStatus = useSelector(state => state.training.status);

  useEffect(() => {
    if (modelId && trainingStatus === 'completed') {
      fetchMetrics();
    }
  }, [modelId, trainingStatus]);

  const fetchMetrics = async () => {
    try {
      const response = await modelService.getMetrics(modelId);
      setMetrics(response.metrics);
    } catch (err) {
      setError('Failed to fetch metrics');
    }
  };

  const handleEvaluate = async () => {
    if (!modelId) {
      setError('No model available for evaluation');
      return;
    }

    try {
      setEvaluating(true);
      setError(null);

      const testData = {
        test_data: [
          { text: "Sample text 1", label: 0 },
          { text: "Sample text 2", label: 1 }
        ]
      };

      const response = await modelService.evaluateModel(modelId, testData);
      setResults(response.results);
    } catch (err) {
      setError(err.message || 'Evaluation failed');
    } finally {
      setEvaluating(false);
    }
  };

  const renderConfusionMatrix = () => {
    if (!results?.confusion_matrix) return null;

    return (
      <TableContainer component={Paper} sx={{ mt: 2 }}>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Predicted/Actual</TableCell>
              <TableCell>Negative</TableCell>
              <TableCell>Positive</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            <TableRow>
              <TableCell>Negative</TableCell>
              <TableCell>{results.confusion_matrix[0][0]}</TableCell>
              <TableCell>{results.confusion_matrix[0][1]}</TableCell>
            </TableRow>
            <TableRow>
              <TableCell>Positive</TableCell>
              <TableCell>{results.confusion_matrix[1][0]}</TableCell>
              <TableCell>{results.confusion_matrix[1][1]}</TableCell>
            </TableRow>
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Model Evaluation
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Box sx={{ mb: 2 }}>
            <Typography variant="body2" sx={{ mb: 1 }}>
              Model ID: {modelId || 'No model selected'}
            </Typography>
            <Typography variant="body2" sx={{ mb: 2 }}>
              Status: {trainingStatus}
            </Typography>

            <Button
              variant="contained"
              onClick={handleEvaluate}
              disabled={evaluating || !modelId || trainingStatus !== 'completed'}
            >
              {evaluating ? <CircularProgress size={24} /> : 'Evaluate Model'}
            </Button>
          </Box>

          {results && (
            <Grid container spacing={2}>
              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Performance Metrics
                    </Typography>
                    <Typography>
                      Accuracy: {(results.accuracy * 100).toFixed(2)}%
                    </Typography>
                    <Typography>
                      Precision: {(results.precision * 100).toFixed(2)}%
                    </Typography>
                    <Typography>
                      Recall: {(results.recall * 100).toFixed(2)}%
                    </Typography>
                    <Typography>
                      F1 Score: {(results.f1_score * 100).toFixed(2)}%
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12} md={6}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      Confusion Matrix
                    </Typography>
                    {renderConfusionMatrix()}
                  </CardContent>
                </Card>
              </Grid>

              <Grid item xs={12}>
                <Card>
                  <CardContent>
                    <Typography variant="h6" gutterBottom>
                      ROC Curve
                    </Typography>
                    {results.roc_curve && (
                      <InteractiveChart
                        data={results.roc_curve}
                        title="ROC Curve"
                        xAxis="fpr"
                        yAxis="tpr"
                        series={[{ name: 'ROC', data: results.roc_curve }]}
                      />
                    )}
                  </CardContent>
                </Card>
              </Grid>
            </Grid>
          )}

          {metrics && metrics.length > 0 && (
            <Box sx={{ mt: 4 }}>
              <Typography variant="h6" gutterBottom>
                Training History
              </Typography>
              <InteractiveChart
                data={metrics}
                title="Training Metrics"
                xAxis="epoch"
                series={['loss', 'accuracy', 'val_loss', 'val_accuracy']}
              />
            </Box>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelEvaluation; 