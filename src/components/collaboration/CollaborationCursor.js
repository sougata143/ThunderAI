import React from 'react';
import { Box, Typography } from '@mui/material';

function CollaborationCursor({ position, user }) {
  return (
    <Box
      sx={{
        position: 'absolute',
        left: position.x,
        top: position.y,
        pointerEvents: 'none',
        zIndex: 1000,
        transform: 'translate(-50%, -50%)'
      }}
    >
      <svg
        width="24"
        height="24"
        viewBox="0 0 24 24"
        fill="none"
        style={{
          transform: 'rotate(-45deg)',
          filter: 'drop-shadow(0 1px 2px rgba(0,0,0,0.3))'
        }}
      >
        <path
          d="M1 1L11 21L14 14L21 11L1 1Z"
          fill={user.color}
          stroke="white"
          strokeWidth="1.5"
        />
      </svg>
      <Typography
        variant="caption"
        sx={{
          position: 'absolute',
          left: 16,
          top: 16,
          background: user.color,
          color: 'white',
          padding: '2px 6px',
          borderRadius: 1,
          whiteSpace: 'nowrap'
        }}
      >
        {user.name}
      </Typography>
    </Box>
  );
}

export default CollaborationCursor; 