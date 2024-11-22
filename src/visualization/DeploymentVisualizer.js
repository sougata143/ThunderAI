import React from 'react';
import {
  Grid,
  Paper,
  Typography,
  Box,
  CircularProgress
} from '@mui/material';
import {
  Timeline,
  TimelineItem,
  TimelineSeparator,
  TimelineConnector,
  TimelineContent,
  TimelineDot
} from '@mui/lab';

function DeploymentVisualizer({ status }) {
  const getStatusColor = (currentStatus) => {
    switch (currentStatus) {
      case 'completed':
        return 'success';
      case 'error':
        return 'error';
      case 'in_progress':
        return 'primary';
      default:
        return 'grey';
    }
  };

  return (
    <Grid container spacing={2}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Deployment Progress
          </Typography>
          <Timeline>
            {status.steps?.map((step, index) => (
              <TimelineItem key={step.name}>
                <TimelineSeparator>
                  <TimelineDot color={getStatusColor(step.status)}>
                    {step.status === 'in_progress' && (
                      <CircularProgress size={20} />
                    )}
                  </TimelineDot>
                  {index < status.steps.length - 1 && <TimelineConnector />}
                </TimelineSeparator>
                <TimelineContent>
                  <Typography>{step.name}</Typography>
                  {step.message && (
                    <Typography variant="body2" color="textSecondary">
                      {step.message}
                    </Typography>
                  )}
                </TimelineContent>
              </TimelineItem>
            ))}
          </Timeline>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default DeploymentVisualizer; 