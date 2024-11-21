import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  CircularProgress,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
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
import { ModelEvaluationService } from '../../services/modelEvaluationService';
import { ModelMetrics, EvaluationResult } from '../../types/evaluation';
import ConfusionMatrixHeatmap from './ConfusionMatrixHeatmap';
import MetricsTable from './MetricsTable';
import ROCCurve from './ROCCurve';
import PRCurve from './PRCurve';

interface Props {
  modelId: string;
}

function ModelEvaluationDashboard({ modelId }: Props) {
  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        setLoading(true);
        const testData = await fetchTestData(); // Implement this function
        const evaluationResult = await ModelEvaluationService.evaluateModel(modelId, testData);
        setMetrics(evaluationResult);
        setError(null);
      } catch (err) {
        setError('Failed to fetch model metrics');
        console.error('Error:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
  }, [modelId]);

  if (loading) {
    return (
      <Grid container justifyContent="center" alignItems="center" style={{ minHeight: '400px' }}>
        <CircularProgress />
      </Grid>
    );
  }

  if (error || !metrics) {
    return (
      <Typography color="error" align="center">
        {error || 'No metrics available'}
      </Typography>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Typography variant="h4" gutterBottom>
          Model Evaluation Dashboard
        </Typography>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Key Metrics
          </Typography>
          <MetricsTable metrics={metrics} />
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Confusion Matrix
          </Typography>
          <ConfusionMatrixHeatmap matrix={metrics.confusionMatrix} />
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            ROC Curve
          </Typography>
          <ROCCurve data={metrics.rocCurve} />
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Precision-Recall Curve
          </Typography>
          <PRCurve data={metrics.prCurve} />
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelEvaluationDashboard; 