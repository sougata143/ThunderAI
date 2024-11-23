import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  Typography,
  Grid,
  CircularProgress,
  Paper,
} from '@mui/material';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import { useParams } from "react-router-dom";
import { useAuth } from "../../contexts/AuthContext";
import { api } from "../../services/api";
import { toast } from "react-hot-toast";

interface Model {
  _id: string;
  name: string;
  description: string;
  metrics?: {
    perplexity: number;
    bleu_score?: number;
    accuracy?: number;
    loss: number;
  };
}

interface MetricCard {
  title: string;
  value: number;
  color: string;
  description: string;
}

export const ModelMetrics: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const { token } = useAuth();
  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await api.get(`/llm/models/${modelId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setModel(response.data);
      } catch (error) {
        toast.error('Failed to fetch model metrics');
        console.error('Error fetching model:', error);
      } finally {
        setLoading(false);
      }
    };

    if (modelId) {
      fetchModel();
    }
  }, [modelId, token]);

  const getMetricCards = (): MetricCard[] => {
    if (!model?.metrics) return [];

    const cards: MetricCard[] = [
      {
        title: 'Perplexity',
        value: model.metrics.perplexity,
        color: '#2196f3',
        description: 'Lower perplexity indicates better model performance',
      },
      {
        title: 'Loss',
        value: model.metrics.loss,
        color: '#f44336',
        description: 'Training loss value',
      },
    ];

    if (model.metrics.accuracy) {
      cards.push({
        title: 'Accuracy',
        value: model.metrics.accuracy,
        color: '#4caf50',
        description: 'Model prediction accuracy',
      });
    }

    if (model.metrics.bleu_score) {
      cards.push({
        title: 'BLEU Score',
        value: model.metrics.bleu_score,
        color: '#ff9800',
        description: 'Text generation quality metric',
      });
    }

    return cards;
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  const metricCards = getMetricCards();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Metrics for {model?.name}
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        {model?.description}
      </Typography>

      <Grid container spacing={3} sx={{ mb: 4 }}>
        {metricCards.map((card) => (
          <Grid item xs={12} sm={6} md={3} key={card.title}>
            <Paper
              sx={{
                p: 3,
                height: '100%',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                textAlign: 'center',
              }}
            >
              <Typography variant="h6" gutterBottom>
                {card.title}
              </Typography>
              <Typography
                variant="h4"
                sx={{ color: card.color, fontWeight: 'bold', my: 2 }}
              >
                {card.value.toFixed(4)}
              </Typography>
              <Typography variant="body2" color="text.secondary">
                {card.description}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>

      <Card>
        <CardContent>
          <Typography variant="h6" gutterBottom>
            Metrics Visualization
          </Typography>
          <Box sx={{ width: '100%', height: 400 }}>
            <ResponsiveContainer>
              <LineChart
                data={[model?.metrics].filter(Boolean)}
                margin={{ top: 5, right: 30, left: 20, bottom: 5 }}
              >
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis />
                <YAxis />
                <Tooltip />
                <Legend />
                {metricCards.map((card) => (
                  <Line
                    key={card.title}
                    type="monotone"
                    dataKey={card.title.toLowerCase()}
                    stroke={card.color}
                    name={card.title}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
};
