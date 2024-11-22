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
import { startTraining, stopTraining, resetTraining } from '../store/slices/trainingSlice';
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
    const dispatch = useDispatch();
    const training = useSelector((state) => state.training);
    const [selectedModel, setSelectedModel] = useState('');
    const [selectedDataset, setSelectedDataset] = useState('');
    const [hyperparameters, setHyperparameters] = useState({
        learningRate: 0.001,
        batchSize: 32,
        epochs: 10
    });

    // Initialize metrics data structure
    const defaultMetrics = {
        loss: [],
        accuracy: [],
        epochs: []
    };

    // Safely access metrics or use default empty arrays
    const metrics = training.metrics || defaultMetrics;

    const chartData = {
        labels: metrics.epochs || [],
        datasets: [
            {
                label: 'Loss',
                data: metrics.loss || [],
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            },
            {
                label: 'Accuracy',
                data: metrics.accuracy || [],
                borderColor: 'rgb(75, 192, 192)',
                tension: 0.1
            }
        ]
    };

    const chartOptions = {
        responsive: true,
        plugins: {
            legend: {
                position: 'top',
            },
            title: {
                display: true,
                text: 'Training Progress'
            }
        },
        scales: {
            y: {
                beginAtZero: true
            }
        }
    };

    const handleStartTraining = async () => {
        const config = {
            model_type: selectedModel,
            dataset_id: selectedDataset,
            learning_rate: hyperparameters.learningRate,
            batch_size: hyperparameters.batchSize,
            epochs: hyperparameters.epochs
        };

        try {
            await dispatch(startTraining(config)).unwrap();
        } catch (err) {
            console.error('Failed to start training:', err);
        }
    };

    const handleStopTraining = async () => {
        if (training.modelId) {
            try {
                await dispatch(stopTraining(training.modelId)).unwrap();
            } catch (err) {
                console.error('Failed to stop training:', err);
            }
        }
    };

    // Helper function to safely render error messages
    const renderError = (error) => {
        if (typeof error === 'string') {
            return error;
        }
        if (error && typeof error === 'object') {
            return error.message || JSON.stringify(error);
        }
        return 'An unknown error occurred';
    };

    return (
        <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
            <Grid container spacing={3}>
                {/* Training Configuration */}
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Training Configuration
                        </Typography>
                        <FormControl fullWidth sx={{ mb: 2 }}>
                            <InputLabel>Model</InputLabel>
                            <Select
                                value={selectedModel}
                                onChange={(e) => setSelectedModel(e.target.value)}
                                label="Model"
                            >
                                <MenuItem value="bert">BERT</MenuItem>
                                <MenuItem value="gpt">GPT</MenuItem>
                                <MenuItem value="lstm">LSTM</MenuItem>
                            </Select>
                        </FormControl>

                        <FormControl fullWidth sx={{ mb: 2 }}>
                            <InputLabel>Dataset</InputLabel>
                            <Select
                                value={selectedDataset}
                                onChange={(e) => setSelectedDataset(e.target.value)}
                                label="Dataset"
                            >
                                <MenuItem value="1">Dataset 1</MenuItem>
                                <MenuItem value="2">Dataset 2</MenuItem>
                            </Select>
                        </FormControl>

                        <TextField
                            fullWidth
                            label="Learning Rate"
                            type="number"
                            value={hyperparameters.learningRate}
                            onChange={(e) => setHyperparameters({
                                ...hyperparameters,
                                learningRate: parseFloat(e.target.value)
                            })}
                            sx={{ mb: 2 }}
                            inputProps={{ step: "0.0001" }}
                        />

                        <TextField
                            fullWidth
                            label="Batch Size"
                            type="number"
                            value={hyperparameters.batchSize}
                            onChange={(e) => setHyperparameters({
                                ...hyperparameters,
                                batchSize: parseInt(e.target.value)
                            })}
                            sx={{ mb: 2 }}
                        />

                        <TextField
                            fullWidth
                            label="Epochs"
                            type="number"
                            value={hyperparameters.epochs}
                            onChange={(e) => setHyperparameters({
                                ...hyperparameters,
                                epochs: parseInt(e.target.value)
                            })}
                            sx={{ mb: 2 }}
                        />

                        <Button
                            variant="contained"
                            fullWidth
                            onClick={handleStartTraining}
                            disabled={training.status === 'training'}
                            sx={{ mb: 1 }}
                        >
                            {training.status === 'training' ? 'Training...' : 'Start Training'}
                        </Button>

                        {training.status === 'training' && (
                            <Button
                                variant="outlined"
                                color="secondary"
                                fullWidth
                                onClick={handleStopTraining}
                            >
                                Stop Training
                            </Button>
                        )}

                        {training.error && (
                            <Alert severity="error" sx={{ mb: 2 }}>
                                {renderError(training.error)}
                            </Alert>
                        )}
                    </Paper>
                </Grid>

                {/* Training Progress */}
                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 2 }}>
                        <Typography variant="h6" gutterBottom>
                            Training Progress
                        </Typography>
                        
                        {training.error && (
                            <Alert severity="error" sx={{ mb: 2 }}>
                                {renderError(training.error)}
                            </Alert>
                        )}

                        {training.status === 'training' && (
                            <Box sx={{ mb: 2 }}>
                                <LinearProgress 
                                    variant="determinate" 
                                    value={training.progress || 0} 
                                />
                                <Typography variant="body2" color="text.secondary" align="center">
                                    {`Epoch ${training.currentEpoch || 0}/${training.totalEpochs || 0}`}
                                </Typography>
                            </Box>
                        )}

                        {metrics && (metrics.loss?.length > 0 || metrics.accuracy?.length > 0) ? (
                            <Box sx={{ height: 400 }}>
                                <Line data={chartData} options={chartOptions} />
                            </Box>
                        ) : (
                            <Typography variant="body1" color="text.secondary" align="center">
                                No training data available
                            </Typography>
                        )}
                    </Paper>
                </Grid>
            </Grid>
        </Container>
    );
};

export default Dashboard; 