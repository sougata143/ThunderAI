import React from 'react';
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  ListItemButton,
  Divider,
  Typography,
  Toolbar,
} from '@mui/material';
import {
  Dashboard as DashboardIcon,
  Psychology as AIIcon,
  Settings as SettingsIcon,
  Person as PersonIcon,
  Science as ExperimentsIcon,
  Logout as LogoutIcon,
} from '@mui/icons-material';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';

const drawerWidth = 240;

interface SidebarProps {
  open: boolean;
  onClose: () => void;
  variant: "permanent" | "temporary";
}

export const Sidebar: React.FC<SidebarProps> = ({ open, onClose, variant }) => {
  const navigate = useNavigate();
  const location = useLocation();
  const { logout } = useAuth();

  const menuItems = [
    { text: 'Dashboard', icon: <DashboardIcon />, path: '/dashboard' },
    { text: 'Experiments', icon: <ExperimentsIcon />, path: '/experiments' },
    { text: 'Language Models', icon: <AIIcon />, path: '/llm' },
    { text: 'Settings', icon: <SettingsIcon />, path: '/settings' },
    { text: 'Profile', icon: <PersonIcon />, path: '/profile' },
  ];

  const handleNavigation = (path: string) => {
    navigate(path);
    if (variant === "temporary") {
      onClose();
    }
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
    if (variant === "temporary") {
      onClose();
    }
  };

  const drawer = (
    <>
      <Toolbar>
        <Typography variant="h6" noWrap component="div">
          ThunderAI
        </Typography>
      </Toolbar>
      <Divider />
      <List>
        {menuItems.map((item) => (
          <ListItem key={item.text} disablePadding>
            <ListItemButton
              selected={location.pathname === item.path}
              onClick={() => handleNavigation(item.path)}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
        <Divider sx={{ my: 1 }} />
        <ListItem disablePadding>
          <ListItemButton onClick={handleLogout}>
            <ListItemIcon>
              <LogoutIcon />
            </ListItemIcon>
            <ListItemText primary="Logout" />
          </ListItemButton>
        </ListItem>
      </List>
    </>
  );

  if (variant === "permanent") {
    return (
      <Drawer
        variant="permanent"
        sx={{
          width: drawerWidth,
          flexShrink: 0,
          '& .MuiDrawer-paper': {
            width: drawerWidth,
            boxSizing: 'border-box',
          },
        }}
      >
        {drawer}
      </Drawer>
    );
  }

  return (
    <Drawer
      variant="temporary"
      open={open}
      onClose={onClose}
      ModalProps={{
        keepMounted: true, // Better open performance on mobile
      }}
      sx={{
        display: { xs: 'block', sm: 'none' },
        '& .MuiDrawer-paper': {
          width: drawerWidth,
          boxSizing: 'border-box',
        },
      }}
    >
      {drawer}
    </Drawer>
  );
};
