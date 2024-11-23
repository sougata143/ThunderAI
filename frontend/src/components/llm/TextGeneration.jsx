import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import {
  Container,
  Paper,
  Typography,
  TextField,
  Button,
  Box,
  Alert,
  Slider,
  FormControl,
  InputLabel,
  Input,
  FormHelperText,
  CircularProgress,
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { api } from '../../services/api';

export const TextGeneration = () => {
  const { modelId } = useParams();
  const { user } = useAuth();
  const [model, setModel] = useState(null);
  const [prompt, setPrompt] = useState('');
  const [parameters, setParameters] = useState({
    max_tokens: 100,
    temperature: 0.7,
    top_p: 0.9,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
  });
  const [generatedText, setGeneratedText] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  useEffect(() => {
    const fetchModel = async () => {
      try {
        const response = await api.get(`/llm/models/${modelId}`);
        setModel(response.data);
      } catch (err) {
        console.error('Error fetching model:', err);
        setError(err.response?.data?.detail || err.message);
      }
    };

    if (modelId && user) {
      fetchModel();
    }
  }, [modelId, user]);

  const handleParameterChange = (param) => (event, newValue) => {
    setParameters((prev) => ({
      ...prev,
      [param]: newValue,
    }));
  };

  const handleGenerate = async () => {
    if (!prompt.trim()) {
      setError('Please enter a prompt');
      return;
    }

    setLoading(true);
    setError('');
    setGeneratedText('');

    try {
      const response = await api.post(`/llm/models/${modelId}/generate`, {
        prompt,
        parameters,
      });
      setGeneratedText(response.data.generated_text);
    } catch (err) {
      console.error('Error generating text:', err);
      setError(err.response?.data?.detail || err.message);
    } finally {
      setLoading(false);
    }
  };

  if (!model) {
    return (
      <Container maxWidth="md" sx={{ mt: 4 }}>
        {error ? (
          <Alert severity="error">{error}</Alert>
        ) : (
          <Box sx={{ display: 'flex', justifyContent: 'center', mt: 4 }}>
            <CircularProgress />
          </Box>
        )}
      </Container>
    );
  }

  return (
    <Container maxWidth="md" sx={{ mt: 4 }}>
      <Paper sx={{ p: 3 }}>
        <Typography variant="h4" gutterBottom>
          Generate Text with {model.name}
        </Typography>

        {error && <Alert severity="error" sx={{ mb: 2 }}>{error}</Alert>}

        <FormControl fullWidth sx={{ mb: 3 }}>
          <TextField
            label="Prompt"
            multiline
            rows={4}
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            variant="outlined"
          />
        </FormControl>

        <Typography variant="h6" gutterBottom>
          Generation Parameters
        </Typography>

        <Box sx={{ mb: 3 }}>
          <Typography gutterBottom>Max Tokens: {parameters.max_tokens}</Typography>
          <Slider
            value={parameters.max_tokens}
            onChange={handleParameterChange('max_tokens')}
            min={1}
            max={2048}
            valueLabelDisplay="auto"
          />

          <Typography gutterBottom>
            Temperature: {parameters.temperature}
          </Typography>
          <Slider
            value={parameters.temperature}
            onChange={handleParameterChange('temperature')}
            min={0}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
          />

          <Typography gutterBottom>Top P: {parameters.top_p}</Typography>
          <Slider
            value={parameters.top_p}
            onChange={handleParameterChange('top_p')}
            min={0}
            max={1}
            step={0.1}
            valueLabelDisplay="auto"
          />

          <Typography gutterBottom>
            Frequency Penalty: {parameters.frequency_penalty}
          </Typography>
          <Slider
            value={parameters.frequency_penalty}
            onChange={handleParameterChange('frequency_penalty')}
            min={-2}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
          />

          <Typography gutterBottom>
            Presence Penalty: {parameters.presence_penalty}
          </Typography>
          <Slider
            value={parameters.presence_penalty}
            onChange={handleParameterChange('presence_penalty')}
            min={-2}
            max={2}
            step={0.1}
            valueLabelDisplay="auto"
          />
        </Box>

        <Box sx={{ mb: 3 }}>
          <Button
            variant="contained"
            color="primary"
            onClick={handleGenerate}
            disabled={loading || !prompt.trim()}
            fullWidth
          >
            {loading ? <CircularProgress size={24} /> : 'Generate'}
          </Button>
        </Box>

        {generatedText && (
          <Box sx={{ mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Generated Text
            </Typography>
            <Paper variant="outlined" sx={{ p: 2 }}>
              <Typography>{generatedText}</Typography>
            </Paper>
          </Box>
        )}
      </Paper>
    </Container>
  );
};
