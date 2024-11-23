import React from 'react';
import { Box, Typography, Paper, Grid, Avatar } from '@mui/material';
import { useAuth } from '../../contexts/AuthContext';

export const Profile: React.FC = () => {
  const { user } = useAuth();

  return (
    <Box sx={{ p: 3 }}>
      <Typography variant="h4" gutterBottom>
        Profile
      </Typography>
      <Grid container spacing={3}>
        <Grid item xs={12}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar sx={{ width: 64, height: 64, mr: 2 }}>
                {user?.name?.[0] || user?.email?.[0]}
              </Avatar>
              <Box>
                <Typography variant="h6">{user?.name || 'User'}</Typography>
                <Typography color="textSecondary">{user?.email}</Typography>
              </Box>
            </Box>
            {/* Add more profile content here */}
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
};
