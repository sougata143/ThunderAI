import React, { useState, useEffect } from 'react';
import {
  Box,
  Card,
  CardContent,
  TextField,
  Button,
  Typography,
  Slider,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Alert,
  Grid,
} from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';
import { API_BASE_URL } from '/src/config.js';

export const SettingsForm = () => {
  const { token } = useAuth();
  const [settings, setSettings] = useState({
    openai_api_key: '',
    default_model: '',
    max_tokens: 1000,
    temperature: 0.7,
    top_p: 1.0,
    frequency_penalty: 0.0,
    presence_penalty: 0.0,
  });
  const [models, setModels] = useState([]);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchSettings();
    fetchModels();
  }, [token]);

  const fetchSettings = async () => {
    try {
      console.log('Token:', token); // Debug token
      const tokenWithPrefix = token.startsWith('Bearer ') ? token : `Bearer ${token}`;
      const response = await fetch(`${API_BASE_URL}/settings`, {
        headers: {
          'Authorization': tokenWithPrefix,
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch settings');
      }
      const data = await response.json();
      setSettings(data);
    } catch (err) {
      console.error('Error fetching settings:', err);
      setError(err.message);
    }
  };

  const fetchModels = async () => {
    try {
      console.log('Token:', token); // Debug token
      const tokenWithPrefix = token.startsWith('Bearer ') ? token : `Bearer ${token}`;
      const response = await fetch(`${API_BASE_URL}/llm/models`, {
        headers: {
          'Authorization': tokenWithPrefix,
          'Content-Type': 'application/json',
        },
      });
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to fetch models');
      }
      const data = await response.json();
      setModels(data);
    } catch (err) {
      console.error('Error fetching models:', err);
      setError(err.message);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError('');
    setSuccess('');

    try {
      const tokenWithPrefix = token.startsWith('Bearer ') ? token : `Bearer ${token}`;
      const response = await fetch(`${API_BASE_URL}/settings`, {
        method: 'PUT',
        headers: {
          'Authorization': tokenWithPrefix,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(settings),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to update settings');
      }

      setSuccess('Settings updated successfully');
    } catch (err) {
      console.error('Error updating settings:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleChange = (field) => (event) => {
    const value = event.target.value;
    setSettings((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  const handleSliderChange = (field) => (event, value) => {
    setSettings((prev) => ({
      ...prev,
      [field]: value,
    }));
  };

  return (
    <Card>
      <CardContent>
        <Typography variant="h5" gutterBottom>
          Settings
        </Typography>

        {error && (
          <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError('')}>
            {error}
          </Alert>
        )}

        {success && (
          <Alert severity="success" sx={{ mb: 2 }} onClose={() => setSuccess('')}>
            {success}
          </Alert>
        )}

        <Box component="form" onSubmit={handleSubmit} sx={{ mt: 2 }}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="OpenAI API Key"
                type="password"
                value={settings.openai_api_key || ''}
                onChange={handleChange('openai_api_key')}
                margin="normal"
              />
            </Grid>

            <Grid item xs={12}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Default Model</InputLabel>
                <Select
                  value={settings.default_model || ''}
                  onChange={handleChange('default_model')}
                  label="Default Model"
                >
                  {models.map((model) => (
                    <MenuItem key={model._id} value={model._id}>
                      {model.name}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Max Tokens: {settings.max_tokens}
              </Typography>
              <Slider
                value={settings.max_tokens}
                onChange={handleSliderChange('max_tokens')}
                min={1}
                max={4000}
                step={1}
                marks={[
                  { value: 1, label: '1' },
                  { value: 2000, label: '2000' },
                  { value: 4000, label: '4000' },
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Temperature: {settings.temperature}
              </Typography>
              <Slider
                value={settings.temperature}
                onChange={handleSliderChange('temperature')}
                min={0}
                max={1}
                step={0.1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Top P: {settings.top_p}
              </Typography>
              <Slider
                value={settings.top_p}
                onChange={handleSliderChange('top_p')}
                min={0}
                max={1}
                step={0.1}
                marks={[
                  { value: 0, label: '0' },
                  { value: 0.5, label: '0.5' },
                  { value: 1, label: '1' },
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Frequency Penalty: {settings.frequency_penalty}
              </Typography>
              <Slider
                value={settings.frequency_penalty}
                onChange={handleSliderChange('frequency_penalty')}
                min={-2}
                max={2}
                step={0.1}
                marks={[
                  { value: -2, label: '-2' },
                  { value: 0, label: '0' },
                  { value: 2, label: '2' },
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Typography gutterBottom>
                Presence Penalty: {settings.presence_penalty}
              </Typography>
              <Slider
                value={settings.presence_penalty}
                onChange={handleSliderChange('presence_penalty')}
                min={-2}
                max={2}
                step={0.1}
                marks={[
                  { value: -2, label: '-2' },
                  { value: 0, label: '0' },
                  { value: 2, label: '2' },
                ]}
              />
            </Grid>

            <Grid item xs={12}>
              <Button
                type="submit"
                variant="contained"
                color="primary"
                disabled={loading}
                fullWidth
              >
                {loading ? 'Saving...' : 'Save Settings'}
              </Button>
            </Grid>
          </Grid>
        </Box>
      </CardContent>
    </Card>
  );
};
