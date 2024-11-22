import React, { useEffect, useState } from 'react';
import {
    Box,
    Paper,
    Typography,
    Button,
    LinearProgress,
    Grid
} from '@mui/material';
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

const TrainingProgress = ({ modelId, onStopTraining }) => {
    const [metrics, setMetrics] = useState({
        loss: [],
        accuracy: [],
        epochs: []
    });
    const [status, setStatus] = useState('starting');
    const [ws, setWs] = useState(null);

    useEffect(() => {
        if (modelId) {
            // Connect to WebSocket
            const wsConnection = new WebSocket(`ws://localhost:8000/api/v1/models/ws/training/${modelId}`);
            
            wsConnection.onopen = () => {
                console.log('WebSocket Connected');
            };

            wsConnection.onmessage = (event) => {
                const data = JSON.parse(event.data);
                if (data.type === 'metrics') {
                    setMetrics(prevMetrics => ({
                        loss: [...prevMetrics.loss, data.loss],
                        accuracy: [...prevMetrics.accuracy, data.accuracy],
                        epochs: [...prevMetrics.epochs, data.epoch]
                    }));
                } else if (data.type === 'status') {
                    setStatus(data.status);
                }
            };

            wsConnection.onerror = (error) => {
                console.error('WebSocket error:', error);
            };

            setWs(wsConnection);

            // Cleanup on unmount
            return () => {
                if (wsConnection) {
                    wsConnection.close();
                }
            };
        }
    }, [modelId]);

    const handleStopTraining = () => {
        if (ws) {
            ws.send('stop');
        }
        if (onStopTraining) {
            onStopTraining();
        }
    };

    const chartData = {
        labels: metrics.epochs,
        datasets: [
            {
                label: 'Loss',
                data: metrics.loss,
                borderColor: 'rgb(255, 99, 132)',
                tension: 0.1
            },
            {
                label: 'Accuracy',
                data: metrics.accuracy,
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

    return (
        <Paper sx={{ p: 3, mt: 3 }}>
            <Box sx={{ mb: 3 }}>
                <Grid container justifyContent="space-between" alignItems="center">
                    <Grid item>
                        <Typography variant="h6" gutterBottom>
                            Training Progress
                        </Typography>
                    </Grid>
                    <Grid item>
                        <Button
                            variant="contained"
                            color="error"
                            onClick={handleStopTraining}
                            disabled={status === 'completed' || status === 'stopped'}
                        >
                            Stop Training
                        </Button>
                    </Grid>
                </Grid>
                <Typography color="textSecondary">
                    Status: {status.charAt(0).toUpperCase() + status.slice(1)}
                </Typography>
                {status === 'starting' && (
                    <Box sx={{ width: '100%', mt: 2 }}>
                        <LinearProgress />
                    </Box>
                )}
            </Box>

            <Box sx={{ height: 400 }}>
                <Line data={chartData} options={chartOptions} />
            </Box>

            {metrics.loss.length > 0 && (
                <Box sx={{ mt: 2 }}>
                    <Typography variant="body2" color="textSecondary">
                        Latest Metrics:
                    </Typography>
                    <Typography>
                        Loss: {metrics.loss[metrics.loss.length - 1]?.toFixed(4)}
                    </Typography>
                    <Typography>
                        Accuracy: {metrics.accuracy[metrics.accuracy.length - 1]?.toFixed(4)}
                    </Typography>
                </Box>
            )}
        </Paper>
    );
};

export default TrainingProgress; 