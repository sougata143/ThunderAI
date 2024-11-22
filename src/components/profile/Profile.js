import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  TextField,
  Button,
  Avatar,
  Box,
  Alert,
  CircularProgress,
  Divider,
  Switch,
  FormControlLabel,
  Card,
  CardContent,
  IconButton
} from '@mui/material';
import {
  Edit as EditIcon,
  Save as SaveIcon,
  Cancel as CancelIcon,
  Security as SecurityIcon,
  Notifications as NotificationsIcon,
  Storage as StorageIcon
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { useAuth } from '../../context/AuthContext';
import { userService } from '../../services/userService';

function Profile() {
  const { user } = useAuth();
  const [editing, setEditing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  const [profileData, setProfileData] = useState({
    username: '',
    email: '',
    firstName: '',
    lastName: '',
    organization: '',
    jobTitle: ''
  });

  const [settings, setSettings] = useState({
    emailNotifications: true,
    modelUpdateAlerts: true,
    experimentNotifications: true,
    twoFactorAuth: false,
    autoSaveExperiments: true,
    darkMode: false
  });

  useEffect(() => {
    if (user) {
      fetchUserProfile();
    }
  }, [user]);

  const fetchUserProfile = async () => {
    try {
      setLoading(true);
      const response = await userService.getProfile();
      setProfileData(response.data);
      setSettings(response.data.settings);
    } catch (err) {
      setError('Failed to load profile data');
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    try {
      setLoading(true);
      setError(null);
      await userService.updateProfile({
        ...profileData,
        settings
      });
      setSuccess('Profile updated successfully');
      setEditing(false);
    } catch (err) {
      setError('Failed to update profile');
    } finally {
      setLoading(false);
    }
  };

  const handleSettingChange = (setting) => (event) => {
    setSettings(prev => ({
      ...prev,
      [setting]: event.target.checked
    }));
  };

  const renderProfileSection = () => (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
          <Avatar
            sx={{ width: 100, height: 100, mr: 2 }}
          >
            {profileData.firstName?.[0] || profileData.username?.[0]}
          </Avatar>
          <Box>
            <Typography variant="h5">
              {profileData.firstName} {profileData.lastName}
            </Typography>
            <Typography color="textSecondary">
              {profileData.jobTitle} at {profileData.organization}
            </Typography>
          </Box>
          <Box sx={{ ml: 'auto' }}>
            {!editing ? (
              <IconButton onClick={() => setEditing(true)}>
                <EditIcon />
              </IconButton>
            ) : (
              <Box>
                <IconButton color="primary" onClick={handleSave}>
                  <SaveIcon />
                </IconButton>
                <IconButton color="error" onClick={() => setEditing(false)}>
                  <CancelIcon />
                </IconButton>
              </Box>
            )}
          </Box>
        </Box>

        <Grid container spacing={2}>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="First Name"
              value={profileData.firstName}
              onChange={(e) => setProfileData({ ...profileData, firstName: e.target.value })}
              disabled={!editing}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Last Name"
              value={profileData.lastName}
              onChange={(e) => setProfileData({ ...profileData, lastName: e.target.value })}
              disabled={!editing}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Email"
              value={profileData.email}
              onChange={(e) => setProfileData({ ...profileData, email: e.target.value })}
              disabled={!editing}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Username"
              value={profileData.username}
              disabled
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Organization"
              value={profileData.organization}
              onChange={(e) => setProfileData({ ...profileData, organization: e.target.value })}
              disabled={!editing}
            />
          </Grid>
          <Grid item xs={12} md={6}>
            <TextField
              fullWidth
              label="Job Title"
              value={profileData.jobTitle}
              onChange={(e) => setProfileData({ ...profileData, jobTitle: e.target.value })}
              disabled={!editing}
            />
          </Grid>
        </Grid>
      </CardContent>
    </Card>
  );

  const renderSettingsSection = () => (
    <Card>
      <CardContent>
        <Typography variant="h6" gutterBottom>
          Settings
        </Typography>
        
        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            <NotificationsIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Notifications
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={settings.emailNotifications}
                onChange={handleSettingChange('emailNotifications')}
              />
            }
            label="Email Notifications"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settings.modelUpdateAlerts}
                onChange={handleSettingChange('modelUpdateAlerts')}
              />
            }
            label="Model Update Alerts"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settings.experimentNotifications}
                onChange={handleSettingChange('experimentNotifications')}
              />
            }
            label="Experiment Notifications"
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box sx={{ mb: 3 }}>
          <Typography variant="subtitle1" gutterBottom>
            <SecurityIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Security
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={settings.twoFactorAuth}
                onChange={handleSettingChange('twoFactorAuth')}
              />
            }
            label="Two-Factor Authentication"
          />
        </Box>

        <Divider sx={{ my: 2 }} />

        <Box>
          <Typography variant="subtitle1" gutterBottom>
            <StorageIcon sx={{ mr: 1, verticalAlign: 'middle' }} />
            Preferences
          </Typography>
          <FormControlLabel
            control={
              <Switch
                checked={settings.autoSaveExperiments}
                onChange={handleSettingChange('autoSaveExperiments')}
              />
            }
            label="Auto-save Experiments"
          />
          <FormControlLabel
            control={
              <Switch
                checked={settings.darkMode}
                onChange={handleSettingChange('darkMode')}
              />
            }
            label="Dark Mode"
          />
        </Box>
      </CardContent>
    </Card>
  );

  if (loading) {
    return (
      <Box display="flex" justifyContent="center" alignItems="center" minHeight="400px">
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h5" gutterBottom>
            Profile
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          {success && (
            <Alert severity="success" sx={{ mb: 2 }}>
              {success}
            </Alert>
          )}

          <Grid container spacing={3}>
            <Grid item xs={12}>
              {renderProfileSection()}
            </Grid>
            <Grid item xs={12}>
              {renderSettingsSection()}
            </Grid>
          </Grid>
        </Paper>
      </Grid>
    </Grid>
  );
}

export default Profile; 