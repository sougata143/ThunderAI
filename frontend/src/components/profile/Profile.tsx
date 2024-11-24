import React, { useState, useEffect } from 'react';
import {
  Box,
  Typography,
  Paper,
  Grid,
  Avatar,
  Button,
  TextField,
  Switch,
  FormControlLabel,
  Divider,
  Alert,
  CircularProgress,
  IconButton,
  Snackbar,
  Card,
  CardContent,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';
import { Edit as EditIcon, Delete as DeleteIcon } from '@mui/icons-material';
import { useAuth } from '../../contexts/AuthContext';
import { FEATURES, ERROR_MESSAGES } from '../../config';
import { profileApi, UserProfile } from '../../api/profile';

const Profile: React.FC = () => {
  const { user, updateUser } = useAuth();
  const [profile, setProfile] = useState<UserProfile | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [editMode, setEditMode] = useState(false);
  const [editedProfile, setEditedProfile] = useState<Partial<UserProfile>>({});
  const [showApiKeyDialog, setShowApiKeyDialog] = useState(false);
  const [newApiKeyName, setNewApiKeyName] = useState('');
  const [showPasswordDialog, setShowPasswordDialog] = useState(false);
  const [passwordData, setPasswordData] = useState({
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
  });

  useEffect(() => {
    if (!loading && user) {
      fetchProfile();
    }
  }, [user, loading]);

  const fetchProfile = async () => {
    try {
      setLoading(true);
      const data = await profileApi.getProfile();
      setProfile(data);
      setEditedProfile(data);
    } catch (err) {
      setError(ERROR_MESSAGES.serverError);
      console.error('Error loading profile:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleEditToggle = () => {
    setEditMode(!editMode);
    if (!editMode) {
      setEditedProfile(profile || {});
    }
  };

  const handleProfileUpdate = async () => {
    try {
      setLoading(true);
      const updated = await profileApi.updateProfile(editedProfile);
      setProfile(updated);
      setEditMode(false);
      setSuccess('Profile updated successfully');
      updateUser(updated);
    } catch (err) {
      setError(ERROR_MESSAGES.serverError);
      console.error('Error updating profile:', err);
    } finally {
      setLoading(false);
    }
  };

  const handlePasswordChange = async () => {
    if (passwordData.newPassword !== passwordData.confirmPassword) {
      setError('Passwords do not match');
      return;
    }
    try {
      await profileApi.changePassword({
        currentPassword: passwordData.currentPassword,
        newPassword: passwordData.newPassword,
      });
      setSuccess('Password changed successfully');
      setShowPasswordDialog(false);
      setPasswordData({ currentPassword: '', newPassword: '', confirmPassword: '' });
    } catch (err) {
      setError(ERROR_MESSAGES.serverError);
      console.error('Error changing password:', err);
    }
  };

  const handleApiKeyGeneration = async () => {
    if (!FEATURES.enableApiKeys) {
      setError('API key feature is not enabled');
      return;
    }
    try {
      const result = await profileApi.generateApiKey(newApiKeyName);
      setSuccess('API key generated successfully');
      setShowApiKeyDialog(false);
      setNewApiKeyName('');
      fetchProfile();  // Refresh profile to show new key
    } catch (err) {
      setError(ERROR_MESSAGES.serverError);
      console.error('Error generating API key:', err);
    }
  };

  const handleApiKeyDeletion = async (keyId: string) => {
    try {
      await profileApi.deleteApiKey(keyId);
      setSuccess('API key deleted successfully');
      fetchProfile();  // Refresh profile to update keys list
    } catch (err) {
      setError(ERROR_MESSAGES.serverError);
      console.error('Error deleting API key:', err);
    }
  };

  const handlePreferencesChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, checked, value } = event.target;
    setEditedProfile(prev => ({
      ...prev,
      preferences: {
        ...prev.preferences,
        [name]: name === 'theme' || name === 'language' ? value : checked,
      },
    }));
  };

  if (loading && !profile) {
    return (
      <Box sx={{ display: 'flex', justifyContent: 'center', p: 3 }}>
        <CircularProgress />
      </Box>
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      <Snackbar
        open={!!error || !!success}
        autoHideDuration={6000}
        onClose={() => { setError(''); setSuccess(''); }}
      >
        <Alert severity={error ? 'error' : 'success'} onClose={() => { setError(''); setSuccess(''); }}>
          {error || success}
        </Alert>
      </Snackbar>

      <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 3 }}>
        <Typography variant="h4">Profile</Typography>
        <Button
          variant="contained"
          startIcon={editMode ? null : <EditIcon />}
          onClick={handleEditToggle}
        >
          {editMode ? 'Cancel' : 'Edit Profile'}
        </Button>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', alignItems: 'center', mb: 3 }}>
              <Avatar
                sx={{ width: 100, height: 100, mr: 2 }}
                src={profile?.avatar}
              >
                {profile?.name?.[0] || profile?.email?.[0]}
              </Avatar>
              <Box>
                {editMode ? (
                  <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                    <TextField
                      label="Name"
                      value={editedProfile.name || ''}
                      onChange={(e) => setEditedProfile({ ...editedProfile, name: e.target.value })}
                    />
                    <TextField
                      label="Job Title"
                      value={editedProfile.jobTitle || ''}
                      onChange={(e) => setEditedProfile({ ...editedProfile, jobTitle: e.target.value })}
                    />
                  </Box>
                ) : (
                  <>
                    <Typography variant="h6">{profile?.name}</Typography>
                    <Typography color="textSecondary">{profile?.email}</Typography>
                    <Typography>{profile?.jobTitle}</Typography>
                  </>
                )}
              </Box>
            </Box>

            <Divider sx={{ my: 3 }} />

            {editMode ? (
              <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                <TextField
                  label="Organization"
                  value={editedProfile.organization || ''}
                  onChange={(e) => setEditedProfile({ ...editedProfile, organization: e.target.value })}
                />
                <TextField
                  label="Bio"
                  multiline
                  rows={4}
                  value={editedProfile.bio || ''}
                  onChange={(e) => setEditedProfile({ ...editedProfile, bio: e.target.value })}
                />
                <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 2 }}>
                  <Button variant="contained" onClick={handleProfileUpdate}>
                    Save Changes
                  </Button>
                </Box>
              </Box>
            ) : (
              <>
                <Typography variant="subtitle1" gutterBottom>Organization</Typography>
                <Typography paragraph>{profile?.organization || 'Not specified'}</Typography>
                <Typography variant="subtitle1" gutterBottom>Bio</Typography>
                <Typography paragraph>{profile?.bio || 'No bio provided'}</Typography>
              </>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Account Settings</Typography>
                {FEATURES.enableNotifications && (
                  <FormControlLabel
                    control={
                      <Switch
                        checked={profile?.preferences?.emailNotifications || false}
                        onChange={handlePreferencesChange}
                        name="emailNotifications"
                        disabled={!editMode}
                      />
                    }
                    label="Email Notifications"
                  />
                )}
                <Box sx={{ mt: 2 }}>
                  <Button
                    variant="outlined"
                    onClick={() => setShowPasswordDialog(true)}
                    fullWidth
                  >
                    Change Password
                  </Button>
                </Box>
              </Paper>
            </Grid>

            {FEATURES.enableApiKeys && (
              <Grid item xs={12}>
                <Paper sx={{ p: 3 }}>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                    <Typography variant="h6">API Keys</Typography>
                    <Button
                      startIcon={<EditIcon />}
                      onClick={() => setShowApiKeyDialog(true)}
                    >
                      New Key
                    </Button>
                  </Box>
                  {profile?.apiKeys?.map((key) => (
                    <Card key={key.id} sx={{ mb: 1 }}>
                      <CardContent sx={{ py: 1, '&:last-child': { pb: 1 } }}>
                        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                          <Box>
                            <Typography variant="subtitle2">{key.name}</Typography>
                            <Typography variant="caption" color="textSecondary">
                              Last used: {new Date(key.lastUsed).toLocaleDateString()}
                            </Typography>
                          </Box>
                          <IconButton
                            size="small"
                            onClick={() => handleApiKeyDeletion(key.id)}
                          >
                            <DeleteIcon />
                          </IconButton>
                        </Box>
                      </CardContent>
                    </Card>
                  ))}
                </Paper>
              </Grid>
            )}

            <Grid item xs={12}>
              <Paper sx={{ p: 3 }}>
                <Typography variant="h6" gutterBottom>Statistics</Typography>
                <Grid container spacing={2}>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">Models Created</Typography>
                    <Typography variant="h6">{profile?.stats?.modelsCreated || 0}</Typography>
                  </Grid>
                  <Grid item xs={6}>
                    <Typography variant="subtitle2">Experiments Run</Typography>
                    <Typography variant="h6">{profile?.stats?.experimentsRun || 0}</Typography>
                  </Grid>
                  <Grid item xs={12}>
                    <Typography variant="subtitle2">Total Training Hours</Typography>
                    <Typography variant="h6">{profile?.stats?.totalTrainingHours || 0}</Typography>
                  </Grid>
                </Grid>
              </Paper>
            </Grid>
          </Grid>
        </Grid>
      </Grid>

      {/* Password Change Dialog */}
      <Dialog open={showPasswordDialog} onClose={() => setShowPasswordDialog(false)}>
        <DialogTitle>Change Password</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, pt: 1 }}>
            <TextField
              label="Current Password"
              type="password"
              value={passwordData.currentPassword}
              onChange={(e) => setPasswordData({ ...passwordData, currentPassword: e.target.value })}
            />
            <TextField
              label="New Password"
              type="password"
              value={passwordData.newPassword}
              onChange={(e) => setPasswordData({ ...passwordData, newPassword: e.target.value })}
            />
            <TextField
              label="Confirm New Password"
              type="password"
              value={passwordData.confirmPassword}
              onChange={(e) => setPasswordData({ ...passwordData, confirmPassword: e.target.value })}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowPasswordDialog(false)}>Cancel</Button>
          <Button onClick={handlePasswordChange} variant="contained">Change Password</Button>
        </DialogActions>
      </Dialog>

      {/* API Key Dialog */}
      <Dialog open={showApiKeyDialog} onClose={() => setShowApiKeyDialog(false)}>
        <DialogTitle>Generate New API Key</DialogTitle>
        <DialogContent>
          <Box sx={{ pt: 1 }}>
            <TextField
              label="Key Name"
              fullWidth
              value={newApiKeyName}
              onChange={(e) => setNewApiKeyName(e.target.value)}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShowApiKeyDialog(false)}>Cancel</Button>
          <Button onClick={handleApiKeyGeneration} variant="contained">Generate</Button>
        </DialogActions>
      </Dialog>
    </Box>
  );
};

export default Profile;
