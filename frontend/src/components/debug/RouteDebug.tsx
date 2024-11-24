import React from 'react';
import { useLocation } from 'react-router-dom';
import { Box, Typography } from '@mui/material';

const RouteDebug: React.FC = () => {
  const location = useLocation();

  return (
    <Box sx={{ p: 3, bgcolor: '#f5f5f5', borderRadius: 1, mb: 2 }}>
      <Typography variant="h6" gutterBottom>Route Debug Info</Typography>
      <Typography>Current Path: {location.pathname}</Typography>
      <Typography>Search: {location.search}</Typography>
      <Typography>Hash: {location.hash}</Typography>
      <Typography>Key: {location.key}</Typography>
      <Typography>State: {JSON.stringify(location.state)}</Typography>
    </Box>
  );
};

export default RouteDebug;
