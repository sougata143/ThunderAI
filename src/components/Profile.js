import React, { useState, useEffect } from 'react';
import {
    Container,
    Paper,
    Typography,
    Box,
    Grid,
    Avatar,
    Divider,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Card,
    CardContent,
    Chip,
    Button,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    TextField
} from '@mui/material';
import {
    Email,
    DateRange,
    WorkHistory,
    Science,
    Edit
} from '@mui/icons-material';
import { useSelector } from 'react-redux';
import { modelService } from '../services/modelService';

const Profile = () => {
    const { user } = useSelector((state) => state.auth);
    const [experiments, setExperiments] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [editDialogOpen, setEditDialogOpen] = useState(false);
    const [editedProfile, setEditedProfile] = useState({
        name: user?.name || '',
        bio: user?.bio || '',
        organization: user?.organization || '',
        role: user?.role || ''
    });

    useEffect(() => {
        fetchUserExperiments();
    }, []);

    const fetchUserExperiments = async () => {
        try {
            setLoading(true);
            const response = await modelService.getExperiments();
            setExperiments(response.data);
            setError(null);
        } catch (err) {
            setError('Failed to fetch experiments');
            console.error('Error fetching experiments:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleEditProfile = () => {
        setEditDialogOpen(true);
    };

    const handleSaveProfile = async () => {
        try {
            // Here you would typically make an API call to update the profile
            // await userService.updateProfile(editedProfile);
            setEditDialogOpen(false);
            // Refresh user data or show success message
        } catch (err) {
            setError('Failed to update profile');
        }
    };

    const getExperimentStats = () => {
        const total = experiments.length;
        const completed = experiments.filter(exp => exp.status === 'completed').length;
        const running = experiments.filter(exp => exp.status === 'running').length;

        return { total, completed, running };
    };

    const stats = getExperimentStats();

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                {/* Profile Overview */}
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, height: '100%' }}>
                        <Box sx={{ display: 'flex', flexDirection: 'column', alignItems: 'center', mb: 3 }}>
                            <Avatar
                                sx={{ width: 120, height: 120, mb: 2 }}
                                src={user?.avatar}
                            />
                            <Typography variant="h5" gutterBottom>
                                {user?.name || user?.email}
                            </Typography>
                            <Button
                                startIcon={<Edit />}
                                onClick={handleEditProfile}
                                variant="outlined"
                                sx={{ mt: 1 }}
                            >
                                Edit Profile
                            </Button>
                        </Box>

                        <List>
                            <ListItem>
                                <ListItemIcon>
                                    <Email />
                                </ListItemIcon>
                                <ListItemText primary="Email" secondary={user?.email} />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon>
                                    <WorkHistory />
                                </ListItemIcon>
                                <ListItemText 
                                    primary="Organization" 
                                    secondary={user?.organization || 'Not specified'} 
                                />
                            </ListItem>
                            <ListItem>
                                <ListItemIcon>
                                    <DateRange />
                                </ListItemIcon>
                                <ListItemText 
                                    primary="Member Since" 
                                    secondary={new Date(user?.createdAt || Date.now()).toLocaleDateString()} 
                                />
                            </ListItem>
                        </List>
                    </Paper>
                </Grid>

                {/* Activity Overview */}
                <Grid item xs={12} md={8}>
                    <Grid container spacing={2}>
                        {/* Stats Cards */}
                        <Grid item xs={12}>
                            <Grid container spacing={2}>
                                <Grid item xs={12} sm={4}>
                                    <Card>
                                        <CardContent>
                                            <Typography color="textSecondary" gutterBottom>
                                                Total Experiments
                                            </Typography>
                                            <Typography variant="h4">
                                                {stats.total}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                                <Grid item xs={12} sm={4}>
                                    <Card>
                                        <CardContent>
                                            <Typography color="textSecondary" gutterBottom>
                                                Completed
                                            </Typography>
                                            <Typography variant="h4">
                                                {stats.completed}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                                <Grid item xs={12} sm={4}>
                                    <Card>
                                        <CardContent>
                                            <Typography color="textSecondary" gutterBottom>
                                                Running
                                            </Typography>
                                            <Typography variant="h4">
                                                {stats.running}
                                            </Typography>
                                        </CardContent>
                                    </Card>
                                </Grid>
                            </Grid>
                        </Grid>

                        {/* Recent Activity */}
                        <Grid item xs={12}>
                            <Paper sx={{ p: 2 }}>
                                <Typography variant="h6" gutterBottom>
                                    Recent Activity
                                </Typography>
                                <List>
                                    {experiments.slice(0, 5).map((experiment) => (
                                        <React.Fragment key={experiment.id}>
                                            <ListItem>
                                                <ListItemIcon>
                                                    <Science />
                                                </ListItemIcon>
                                                <ListItemText
                                                    primary={experiment.name}
                                                    secondary={`Model: ${experiment.modelType}`}
                                                />
                                                <Chip
                                                    label={experiment.status}
                                                    color={
                                                        experiment.status === 'completed' ? 'success' :
                                                        experiment.status === 'running' ? 'primary' :
                                                        'default'
                                                    }
                                                    size="small"
                                                />
                                            </ListItem>
                                            <Divider />
                                        </React.Fragment>
                                    ))}
                                </List>
                            </Paper>
                        </Grid>
                    </Grid>
                </Grid>
            </Grid>

            {/* Edit Profile Dialog */}
            <Dialog open={editDialogOpen} onClose={() => setEditDialogOpen(false)}>
                <DialogTitle>Edit Profile</DialogTitle>
                <DialogContent>
                    <Box sx={{ pt: 2 }}>
                        <TextField
                            fullWidth
                            label="Name"
                            value={editedProfile.name}
                            onChange={(e) => setEditedProfile({
                                ...editedProfile,
                                name: e.target.value
                            })}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Bio"
                            multiline
                            rows={4}
                            value={editedProfile.bio}
                            onChange={(e) => setEditedProfile({
                                ...editedProfile,
                                bio: e.target.value
                            })}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Organization"
                            value={editedProfile.organization}
                            onChange={(e) => setEditedProfile({
                                ...editedProfile,
                                organization: e.target.value
                            })}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Role"
                            value={editedProfile.role}
                            onChange={(e) => setEditedProfile({
                                ...editedProfile,
                                role: e.target.value
                            })}
                        />
                    </Box>
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setEditDialogOpen(false)}>Cancel</Button>
                    <Button onClick={handleSaveProfile} variant="contained">Save</Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
};

export default Profile; 