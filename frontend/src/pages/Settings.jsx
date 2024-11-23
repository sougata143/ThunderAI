import React from 'react';
import { Container, Typography, Box } from '@mui/material';
import { SettingsForm } from '../components/settings/SettingsForm';

export const Settings = () => {
  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" component="h1" gutterBottom>
          Settings
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Configure your LLM model settings and API preferences.
        </Typography>
      </Box>
      <SettingsForm />
    </Container>
  );
};
