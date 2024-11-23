import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Typography,
  Slider,
  Button,
  Paper,
  CircularProgress,
} from '@mui/material';
import { LoadingButton } from '@mui/lab';
import { useParams } from "react-router-dom";
import { useAuth } from "../../contexts/AuthContext";
import { api } from "../../services/api";
import { toast } from "react-hot-toast";

interface Model {
  _id: string;
  name: string;
  description: string;
}

export const TextGeneration: React.FC = () => {
  const { modelId } = useParams<{ modelId: string }>();
  const { token } = useAuth();
  const [model, setModel] = useState<Model | null>(null);
  const [loading, setLoading] = useState(false);
  const [generating, setGenerating] = useState(false);
  const [formData, setFormData] = useState({
    prompt: '',
    max_length: 100,
    temperature: 0.7,
    top_p: 0.9,
  });
  const [generatedText, setGeneratedText] = useState('');

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await api.get(`/llm/models/${modelId}`, {
          headers: { Authorization: `Bearer ${token}` },
        });
        setModel(response.data);
      } catch (error) {
        toast.error('Failed to fetch model details');
        console.error('Error fetching model:', error);
      } finally {
        setLoading(false);
      }
    };

    if (modelId) {
      setLoading(true);
      fetchModel();
    }
  }, [modelId, token]);

  const handleGenerate = async () => {
    setGenerating(true);
    try {
      const response = await api.post(
        `/llm/models/${modelId}/generate`,
        formData,
        {
          headers: { Authorization: `Bearer ${token}` },
        }
      );
      setGeneratedText(response.data.generated_text);
      toast.success('Text generated successfully');
    } catch (error) {
      toast.error('Failed to generate text');
      console.error('Error generating text:', error);
    } finally {
      setGenerating(false);
    }
  };

  const handleInputChange = (field: string) => (
    event: React.ChangeEvent<HTMLInputElement> | any
  ) => {
    const value = event.target ? event.target.value : event;
    setFormData((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="200px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Generate Text with {model?.name}
      </Typography>
      <Typography variant="body1" color="text.secondary" paragraph>
        {model?.description}
      </Typography>

      <Card sx={{ mb: 3 }}>
        <CardContent>
          <TextField
            label="Prompt"
            multiline
            rows={4}
            fullWidth
            value={formData.prompt}
            onChange={handleInputChange('prompt')}
            sx={{ mb: 3 }}
          />

          <Typography gutterBottom>Max Length: {formData.max_length}</Typography>
          <Slider
            value={formData.max_length}
            onChange={(_, value) => handleInputChange('max_length')(value)}
            min={10}
            max={1000}
            step={10}
            valueLabelDisplay="auto"
            sx={{ mb: 3 }}
          />

          <Typography gutterBottom>Temperature: {formData.temperature}</Typography>
          <Slider
            value={formData.temperature}
            onChange={(_, value) => handleInputChange('temperature')(value)}
            min={0.1}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
            sx={{ mb: 3 }}
          />

          <Typography gutterBottom>Top P: {formData.top_p}</Typography>
          <Slider
            value={formData.top_p}
            onChange={(_, value) => handleInputChange('top_p')(value)}
            min={0.1}
            max={1}
            step={0.1}
            valueLabelDisplay="auto"
            sx={{ mb: 3 }}
          />

          <LoadingButton
            variant="contained"
            onClick={handleGenerate}
            loading={generating}
            disabled={!formData.prompt}
            fullWidth
          >
            Generate
          </LoadingButton>
        </CardContent>
      </Card>

      {generatedText && (
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Generated Text
          </Typography>
          <Typography variant="body1" sx={{ whiteSpace: 'pre-wrap' }}>
            {generatedText}
          </Typography>
        </Paper>
      )}
    </Box>
  );
};
