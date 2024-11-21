import React from 'react';
import {
  Box,
  Avatar,
  Tooltip,
  Badge,
  styled,
  Paper,
  Typography
} from '@mui/material';

const OnlineBadge = styled(Badge)(({ theme }) => ({
  '& .MuiBadge-badge': {
    backgroundColor: '#44b700',
    color: '#44b700',
    boxShadow: `0 0 0 2px ${theme.palette.background.paper}`,
    '&::after': {
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      borderRadius: '50%',
      animation: 'ripple 1.2s infinite ease-in-out',
      border: '1px solid currentColor',
      content: '""',
    },
  },
  '@keyframes ripple': {
    '0%': {
      transform: 'scale(.8)',
      opacity: 1,
    },
    '100%': {
      transform: 'scale(2.4)',
      opacity: 0,
    },
  },
}));

function Collaborators({ users }) {
  return (
    <Paper 
      elevation={3}
      sx={{
        position: 'fixed',
        bottom: 16,
        right: 16,
        p: 2,
        zIndex: 1000,
        display: 'flex',
        flexDirection: 'column',
        gap: 1
      }}
    >
      <Typography variant="subtitle2" gutterBottom>
        Active Collaborators
      </Typography>
      <Box display="flex" gap={1}>
        {users.map((user) => (
          <Tooltip key={user.id} title={user.name}>
            <OnlineBadge
              overlap="circular"
              anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
              variant="dot"
            >
              <Avatar
                alt={user.name}
                src={user.avatar}
                sx={{ width: 32, height: 32 }}
              >
                {user.name.charAt(0)}
              </Avatar>
            </OnlineBadge>
          </Tooltip>
        ))}
      </Box>
    </Paper>
  );
}

export default Collaborators; 