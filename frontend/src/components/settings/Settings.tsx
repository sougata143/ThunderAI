import React from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Switch,
  FormControlLabel,
  Divider,
} from '@mui/material';
import { FEATURES } from '../../config/index';

const Settings: React.FC = () => {
  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12} md={6}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Features
            </Typography>
            <FormControlLabel
              control={<Switch checked={FEATURES.enableApiKeys} />}
              label="API Keys Management"
            />
            <FormControlLabel
              control={<Switch checked={FEATURES.enable2FA} />}
              label="Two-Factor Authentication"
            />
            <FormControlLabel
              control={<Switch checked={FEATURES.enableNotifications} />}
              label="System Notifications"
            />
            <FormControlLabel
              control={<Switch checked={FEATURES.enableModelMetrics} />}
              label="Model Performance Metrics"
            />
            <FormControlLabel
              control={<Switch checked={FEATURES.enableExperiments} />}
              label="Experiments"
            />
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};

export default Settings;
