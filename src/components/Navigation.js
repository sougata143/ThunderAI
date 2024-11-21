import React from 'react';
import { AppBar, Toolbar, Typography, Button, Box, Chip } from '@mui/material';
import { Link, useNavigate } from 'react-router-dom';
import { useSelector, useDispatch } from 'react-redux';
import { logout } from '../store/slices/authSlice';

function Navigation() {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { isAuthenticated, isGuest } = useSelector(state => state.auth);

  const handleNavigation = (path) => {
    navigate(path);
  };

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography variant="h6" component={Link} to="/" style={{ textDecoration: 'none', color: 'white' }}>
          ThunderAI
        </Typography>
        <Box sx={{ flexGrow: 1 }} />
        {isAuthenticated && (
          <Button color="inherit" component={Link} to="/dashboard">
            Dashboard
          </Button>
        )}
        {isAuthenticated && !isGuest && (
          <>
            <Button color="inherit" component={Link} to="/experiments">
              Experiments
            </Button>
            <Button color="inherit" component={Link} to="/model">
              Models
            </Button>
          </>
        )}
        {isAuthenticated ? (
          <>
            {isGuest && (
              <Chip
                label="Guest Mode"
                color="secondary"
                size="small"
                sx={{ mr: 2 }}
              />
            )}
            <Button color="inherit" onClick={handleLogout}>
              {isGuest ? 'Exit Guest Mode' : 'Logout'}
            </Button>
          </>
        ) : (
          <Button color="inherit" onClick={() => handleNavigation('/login')}>
            Login
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
}

export default Navigation; 