import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDispatch, useSelector } from 'react-redux';
import { logout } from '../store/slices/authSlice';
import {
  AppBar,
  Toolbar,
  Typography,
  Button,
  Box,
  IconButton,
  Menu,
  MenuItem,
  Divider,
  Avatar,
  ListItemIcon
} from '@mui/material';
import {
  AccountCircle,
  Dashboard,
  Science,
  Assessment,
  CloudUpload,
  Logout,
  ModelTraining,
  MonitorHeart
} from '@mui/icons-material';

function Navigation() {
  const navigate = useNavigate();
  const dispatch = useDispatch();
  const { user, token } = useSelector((state) => state.auth);
  
  const [anchorEl, setAnchorEl] = useState(null);
  const [modelMenuAnchor, setModelMenuAnchor] = useState(null);

  const handleMenu = (event) => {
    setAnchorEl(event.currentTarget);
  };

  const handleModelMenu = (event) => {
    setModelMenuAnchor(event.currentTarget);
  };

  const handleClose = () => {
    setAnchorEl(null);
  };

  const handleModelMenuClose = () => {
    setModelMenuAnchor(null);
  };

  const handleLogout = () => {
    dispatch(logout());
    navigate('/login');
    handleClose();
  };

  const handleNavigate = (path) => {
    navigate(path);
    handleModelMenuClose();
  };

  return (
    <AppBar position="static">
      <Toolbar>
        <Typography 
          variant="h6" 
          component="div" 
          sx={{ flexGrow: 0, mr: 2, cursor: 'pointer' }}
          onClick={() => navigate('/dashboard')}
        >
          ThunderAI
        </Typography>

        {token && (
          <>
            <Button
              color="inherit"
              startIcon={<Dashboard />}
              onClick={() => navigate('/dashboard')}
              sx={{ mr: 2 }}
            >
              Dashboard
            </Button>

            <Button
              color="inherit"
              startIcon={<Science />}
              onClick={() => navigate('/experiments')}
              sx={{ mr: 2 }}
            >
              Experiments
            </Button>

            <Button
              color="inherit"
              startIcon={<ModelTraining />}
              onClick={handleModelMenu}
              sx={{ mr: 2 }}
            >
              Models
            </Button>
            <Menu
              anchorEl={modelMenuAnchor}
              open={Boolean(modelMenuAnchor)}
              onClose={handleModelMenuClose}
            >
              <MenuItem onClick={() => handleNavigate('/model/training')}>
                <ListItemIcon>
                  <ModelTraining fontSize="small" />
                </ListItemIcon>
                Training
              </MenuItem>
              <MenuItem onClick={() => handleNavigate('/model/evaluation')}>
                <ListItemIcon>
                  <Assessment fontSize="small" />
                </ListItemIcon>
                Evaluation
              </MenuItem>
              <MenuItem onClick={() => handleNavigate('/model/deployment')}>
                <ListItemIcon>
                  <CloudUpload fontSize="small" />
                </ListItemIcon>
                Deployment
              </MenuItem>
              <MenuItem onClick={() => handleNavigate('/model/monitoring')}>
                <ListItemIcon>
                  <MonitorHeart fontSize="small" />
                </ListItemIcon>
                Monitoring
              </MenuItem>
            </Menu>
          </>
        )}

        <Box sx={{ flexGrow: 1 }} />

        {token ? (
          <div>
            <IconButton
              size="large"
              aria-label="account of current user"
              aria-controls="menu-appbar"
              aria-haspopup="true"
              onClick={handleMenu}
              color="inherit"
            >
              <Avatar sx={{ width: 32, height: 32 }}>
                {user?.username?.[0]?.toUpperCase() || <AccountCircle />}
              </Avatar>
            </IconButton>
            <Menu
              id="menu-appbar"
              anchorEl={anchorEl}
              anchorOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              keepMounted
              transformOrigin={{
                vertical: 'top',
                horizontal: 'right',
              }}
              open={Boolean(anchorEl)}
              onClose={handleClose}
            >
              <MenuItem onClick={() => {
                handleClose();
                navigate('/profile');
              }}>
                <ListItemIcon>
                  <AccountCircle fontSize="small" />
                </ListItemIcon>
                Profile
              </MenuItem>
              <Divider />
              <MenuItem onClick={handleLogout}>
                <ListItemIcon>
                  <Logout fontSize="small" />
                </ListItemIcon>
                Logout
              </MenuItem>
            </Menu>
          </div>
        ) : (
          <Button color="inherit" onClick={() => navigate('/login')}>
            Login
          </Button>
        )}
      </Toolbar>
    </AppBar>
  );
}

export default Navigation; 