import React, { useState, useEffect } from 'react';
import {
    Container,
    Paper,
    Typography,
    Grid,
    Card,
    CardContent,
    CardActions,
    Button,
    Box,
    Chip,
    LinearProgress,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    IconButton,
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions
} from '@mui/material';
import {
    PlayArrow,
    Stop,
    Delete,
    Compare,
    Info,
    Download
} from '@mui/icons-material';
import { Line } from 'react-chartjs-2';
import { useDispatch, useSelector } from 'react-redux';
import { modelService } from '../services/modelService';

const Experiments = () => {
    const [experiments, setExperiments] = useState([]);
    const [selectedExperiment, setSelectedExperiment] = useState(null);
    const [dialogOpen, setDialogOpen] = useState(false);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);

    const dispatch = useDispatch();
    const { user } = useSelector((state) => state.auth);

    useEffect(() => {
        fetchExperiments();
    }, []);

    const fetchExperiments = async () => {
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

    const handleStartExperiment = async (experimentId) => {
        try {
            await modelService.startExperiment(experimentId);
            fetchExperiments(); // Refresh list
        } catch (err) {
            setError('Failed to start experiment');
        }
    };

    const handleStopExperiment = async (experimentId) => {
        try {
            await modelService.stopExperiment(experimentId);
            fetchExperiments(); // Refresh list
        } catch (err) {
            setError('Failed to stop experiment');
        }
    };

    const handleDeleteExperiment = async (experimentId) => {
        try {
            await modelService.deleteExperiment(experimentId);
            fetchExperiments(); // Refresh list
        } catch (err) {
            setError('Failed to delete experiment');
        }
    };

    const handleExportResults = async (experimentId) => {
        try {
            await modelService.exportExperimentResults(experimentId);
        } catch (err) {
            setError('Failed to export results');
        }
    };

    const handleCompare = (experiment) => {
        setSelectedExperiment(experiment);
        setDialogOpen(true);
    };

    const getStatusColor = (status) => {
        const colors = {
            'running': 'primary',
            'completed': 'success',
            'failed': 'error',
            'stopped': 'warning'
        };
        return colors[status] || 'default';
    };

    const chartOptions = {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                type: 'linear'
            }
        },
        plugins: {
            legend: {
                position: 'top',
            }
        }
    };

    if (loading) {
        return (
            <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
                <LinearProgress />
            </Container>
        );
    }

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Typography variant="h4" gutterBottom>
                Experiments
            </Typography>

            {error && (
                <Box sx={{ mb: 2 }}>
                    <Typography color="error">{error}</Typography>
                </Box>
            )}

            <Grid container spacing={3}>
                {experiments.map((experiment) => (
                    <Grid item xs={12} key={experiment.id}>
                        <Card>
                            <CardContent>
                                <Grid container spacing={2}>
                                    <Grid item xs={12} md={4}>
                                        <Typography variant="h6">
                                            {experiment.name}
                                        </Typography>
                                        <Typography color="textSecondary" gutterBottom>
                                            Model: {experiment.modelType}
                                        </Typography>
                                        <Chip 
                                            label={experiment.status}
                                            color={getStatusColor(experiment.status)}
                                            size="small"
                                            sx={{ mt: 1 }}
                                        />
                                    </Grid>
                                    <Grid item xs={12} md={8}>
                                        <Box sx={{ height: 200 }}>
                                            <Line
                                                data={{
                                                    labels: experiment.metrics.epochs,
                                                    datasets: [
                                                        {
                                                            label: 'Loss',
                                                            data: experiment.metrics.loss,
                                                            borderColor: 'rgb(255, 99, 132)',
                                                            tension: 0.1
                                                        },
                                                        {
                                                            label: 'Accuracy',
                                                            data: experiment.metrics.accuracy,
                                                            borderColor: 'rgb(75, 192, 192)',
                                                            tension: 0.1
                                                        }
                                                    ]
                                                }}
                                                options={chartOptions}
                                            />
                                        </Box>
                                    </Grid>
                                </Grid>
                            </CardContent>
                            <CardActions>
                                {experiment.status === 'running' ? (
                                    <IconButton 
                                        onClick={() => handleStopExperiment(experiment.id)}
                                        color="warning"
                                    >
                                        <Stop />
                                    </IconButton>
                                ) : (
                                    <IconButton 
                                        onClick={() => handleStartExperiment(experiment.id)}
                                        color="primary"
                                        disabled={experiment.status === 'completed'}
                                    >
                                        <PlayArrow />
                                    </IconButton>
                                )}
                                <IconButton 
                                    onClick={() => handleCompare(experiment)}
                                    color="primary"
                                >
                                    <Compare />
                                </IconButton>
                                <IconButton 
                                    onClick={() => handleExportResults(experiment.id)}
                                    color="primary"
                                >
                                    <Download />
                                </IconButton>
                                <IconButton 
                                    onClick={() => handleDeleteExperiment(experiment.id)}
                                    color="error"
                                >
                                    <Delete />
                                </IconButton>
                            </CardActions>
                        </Card>
                    </Grid>
                ))}
            </Grid>

            <Dialog
                open={dialogOpen}
                onClose={() => setDialogOpen(false)}
                maxWidth="md"
                fullWidth
            >
                <DialogTitle>
                    Experiment Details
                </DialogTitle>
                <DialogContent>
                    {selectedExperiment && (
                        <TableContainer component={Paper}>
                            <Table>
                                <TableHead>
                                    <TableRow>
                                        <TableCell>Metric</TableCell>
                                        <TableCell align="right">Value</TableCell>
                                    </TableRow>
                                </TableHead>
                                <TableBody>
                                    <TableRow>
                                        <TableCell>Final Loss</TableCell>
                                        <TableCell align="right">
                                            {selectedExperiment.metrics.loss.slice(-1)[0]?.toFixed(4)}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Final Accuracy</TableCell>
                                        <TableCell align="right">
                                            {selectedExperiment.metrics.accuracy.slice(-1)[0]?.toFixed(4)}
                                        </TableCell>
                                    </TableRow>
                                    <TableRow>
                                        <TableCell>Training Time</TableCell>
                                        <TableCell align="right">
                                            {selectedExperiment.trainingTime} minutes
                                        </TableCell>
                                    </TableRow>
                                </TableBody>
                            </Table>
                        </TableContainer>
                    )}
                </DialogContent>
                <DialogActions>
                    <Button onClick={() => setDialogOpen(false)}>Close</Button>
                </DialogActions>
            </Dialog>
        </Container>
    );
};

export default Experiments; 