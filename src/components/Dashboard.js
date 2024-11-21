import React, { useState, useEffect, useRef } from 'react';
import { useDispatch, useSelector } from 'react-redux';
import {
    Container,
    Typography,
    Box,
    Paper,
    Grid,
    Chip,
    Select,
    MenuItem,
    TextField,
    Button,
    LinearProgress,
    FormControl,
    InputLabel,
    Card,
    CardContent,
    Alert,
    Divider
} from '@mui/material';
import {
    Timeline,
    TimelineItem,
    TimelineSeparator,
    TimelineConnector,
    TimelineContent,
    TimelineDot
} from '@mui/lab';
import { Line } from 'react-chartjs-2';
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
} from 'chart.js';
import { startTraining, stopTraining, updateMetrics, updateProgress } from '../store/slices/trainingSlice';
import { wsService } from '../services/websocketService';

// Register ChartJS components
ChartJS.register(
    CategoryScale,
    LinearScale,
    PointElement,
    LineElement,
    Title,
    Tooltip,
    Legend
);

const Dashboard = () => {
    const { user, isGuest } = useSelector((state) => state.auth);
    const [selectedModel, setSelectedModel] = useState('');
    const [trainingParams, setTrainingParams] = useState({
        learningRate: 0.001,
        batchSize: 32,
        epochs: 10,
        validationSplit: 0.2
    });
    const [error, setError] = useState(null);
    const dispatch = useDispatch();
    const training = useSelector((state) => state.training);
    const chartRef = useRef(null);

    // Available models
    const models = [
        { id: 'bert', name: 'BERT' },
        { id: 'gpt', name: 'GPT' },
        { id: 'transformer', name: 'Custom Transformer' },
        { id: 'lstm', name: 'LSTM' }
    ];

    useEffect(() => {
        if (training.currentModel) {
            // Connect to WebSocket for real-time updates
            wsService.connect(training.currentModel);
            
            const subscriptionId = wsService.subscribe((data) => {
                if (data.metrics) {
                    dispatch(updateMetrics(data.metrics));
                }
                if (data.progress) {
                    dispatch(updateProgress(data.progress));
                }
            });

            return () => {
                wsService.unsubscribe(subscriptionId);
                wsService.disconnect();
                
                // Cleanup chart
                if (chartRef.current) {
                    chartRef.current.destroy();
                }
            };
        }
    }, [training.currentModel, dispatch]);

    const handleStartTraining = async () => {
        if (!selectedModel) {
            setError('Please select a model first');
            return;
        }

        try {
            const modelConfig = {
                modelType: selectedModel,
                params: {
                    learningRate: parseFloat(trainingParams.learningRate),
                    batchSize: parseInt(trainingParams.batchSize),
                    epochs: parseInt(trainingParams.epochs),
                    validationSplit: parseFloat(trainingParams.validationSplit)
                }
            };

            await dispatch(startTraining(modelConfig)).unwrap();
            setError(null);
        } catch (err) {
            setError(err.message || 'Failed to start training');
        }
    };

    const handleStopTraining = async () => {
        if (training.currentModel) {
            try {
                await dispatch(stopTraining(training.currentModel)).unwrap();
            } catch (err) {
                setError(err.message || 'Failed to stop training');
            }
        }
    };

    const chartOptions = {
        responsive: true,
        scales: {
            y: {
                beginAtZero: true,
                type: 'linear'
            },
            x: {
                type: 'linear',
                beginAtZero: true
            }
        },
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Training Metrics'
            }
        },
        maintainAspectRatio: false
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            {/* Header Section */}
            <Box sx={{ mb: 4 }}>
                <Typography variant="h4" gutterBottom>
                    Welcome{user ? `, ${user.email}` : ''}!
                </Typography>
                {isGuest && (
                    <Chip
                        label="Guest User"
                        color="primary"
                        variant="outlined"
                        sx={{ mb: 2 }}
                    />
                )}
            </Box>

            <Grid container spacing={3}>
                {/* Model Selection and Parameters */}
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2, height: '100%' }}>
                        <Typography variant="h6" gutterBottom>
                            Model Configuration
                        </Typography>
                        <FormControl fullWidth sx={{ mb: 2 }}>
                            <InputLabel>Select Model</InputLabel>
                            <Select
                                value={selectedModel}
                                label="Select Model"
                                onChange={(e) => setSelectedModel(e.target.value)}
                            >
                                {models.map(model => (
                                    <MenuItem key={model.id} value={model.id}>
                                        {model.name}
                                    </MenuItem>
                                ))}
                            </Select>
                        </FormControl>

                        <Typography variant="subtitle2" gutterBottom>
                            Training Parameters
                        </Typography>
                        <TextField
                            fullWidth
                            label="Learning Rate"
                            type="number"
                            value={trainingParams.learningRate}
                            onChange={(e) => setTrainingParams(prev => ({
                                ...prev,
                                learningRate: parseFloat(e.target.value)
                            }))}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Batch Size"
                            type="number"
                            value={trainingParams.batchSize}
                            onChange={(e) => setTrainingParams(prev => ({
                                ...prev,
                                batchSize: parseInt(e.target.value)
                            }))}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Epochs"
                            type="number"
                            value={trainingParams.epochs}
                            onChange={(e) => setTrainingParams(prev => ({
                                ...prev,
                                epochs: parseInt(e.target.value)
                            }))}
                            sx={{ mb: 2 }}
                        />
                        <TextField
                            fullWidth
                            label="Validation Split"
                            type="number"
                            value={trainingParams.validationSplit}
                            onChange={(e) => setTrainingParams(prev => ({
                                ...prev,
                                validationSplit: parseFloat(e.target.value)
                            }))}
                            sx={{ mb: 2 }}
                        />

                        {error && (
                            <Alert severity="error" sx={{ mb: 2 }}>
                                {error}
                            </Alert>
                        )}

                        <Button
                            fullWidth
                            variant="contained"
                            color={training.status === 'training' ? "error" : "primary"}
                            onClick={training.status === 'training' ? handleStopTraining : handleStartTraining}
                        >
                            {training.status === 'training' ? "Stop Training" : "Start Training"}
                        </Button>
                    </Paper>
                </Grid>

                {/* Training Progress */}
                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 2, height: '100%' }}>
                        <Typography variant="h6" gutterBottom>
                            Training Progress
                        </Typography>
                        
                        {training.status === 'training' && (
                            <>
                                <Box sx={{ mb: 2 }}>
                                    <Typography variant="body2" color="textSecondary">
                                        Progress: {training.progress}%
                                    </Typography>
                                    <LinearProgress 
                                        variant="determinate" 
                                        value={training.progress} 
                                        sx={{ mt: 1 }}
                                    />
                                </Box>

                                <Timeline>
                                    <TimelineItem>
                                        <TimelineSeparator>
                                            <TimelineDot color="primary" />
                                            <TimelineConnector />
                                        </TimelineSeparator>
                                        <TimelineContent>
                                            <Typography>
                                                Current Epoch: {training.currentEpoch}/{trainingParams.epochs}
                                            </Typography>
                                        </TimelineContent>
                                    </TimelineItem>
                                    <TimelineItem>
                                        <TimelineSeparator>
                                            <TimelineDot color="primary" />
                                        </TimelineSeparator>
                                        <TimelineContent>
                                            <Typography>
                                                Latest Loss: {training.metrics.loss[training.metrics.loss.length - 1]?.toFixed(4) || 0}
                                            </Typography>
                                        </TimelineContent>
                                    </TimelineItem>
                                </Timeline>

                                {/* Training Metrics Visualization */}
                                <Box sx={{ mt: 2, height: '400px' }}>
                                    <Typography variant="subtitle1" gutterBottom>
                                        Training Metrics
                                    </Typography>
                                    <Line
                                        ref={chartRef}
                                        data={{
                                            labels: Array.from({ length: training.metrics.loss.length }, (_, i) => i + 1),
                                            datasets: [
                                                {
                                                    label: 'Loss',
                                                    data: training.metrics.loss,
                                                    borderColor: 'rgb(255, 99, 132)',
                                                    tension: 0.1
                                                },
                                                {
                                                    label: 'Accuracy',
                                                    data: training.metrics.accuracy,
                                                    borderColor: 'rgb(75, 192, 192)',
                                                    tension: 0.1
                                                }
                                            ]
                                        }}
                                        options={chartOptions}
                                    />
                                </Box>
                            </>
                        )}

                        {!training.status && !selectedModel && (
                            <Typography color="textSecondary" align="center">
                                Select a model and configure parameters to start training
                            </Typography>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default Dashboard; 