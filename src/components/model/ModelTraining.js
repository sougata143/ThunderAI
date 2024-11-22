import React, { useState } from 'react';
import {
    Container,
    Paper,
    Typography,
    Box,
    Grid,
    TextField,
    Button,
    FormControl,
    InputLabel,
    Select,
    MenuItem,
    Alert,
    CircularProgress,
    Chip,
    FormControlLabel,
    Switch
} from '@mui/material';
import { modelService } from '../../services/modelService';
import ErrorBoundary from '../common/ErrorBoundary';
import TrainingProgress from './TrainingProgress';

const ModelTraining = () => {
    const initialConfig = {
        modelType: 'bert',
        optimizer: 'adam',
        loss: 'categorical_crossentropy',
        metrics: ['accuracy'],
        validationSplit: 0.2,
        shuffle: true,
        verbose: 1,
        batchSize: 32,
        epochs: 10,
        learningRate: 0.001,
        datasetPath: '',  // Add any additional fields your API expects
        outputPath: ''
    };

    const [trainingConfig, setTrainingConfig] = useState(initialConfig);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);
    const [success, setSuccess] = useState(null);
    const [modelId, setModelId] = useState(null);

    const handleInputChange = (field) => (event) => {
        let value = event.target.value;
        
        // Convert numeric fields
        if (['batchSize', 'epochs'].includes(field)) {
            value = parseInt(value, 10);
        } else if (['learningRate', 'validationSplit'].includes(field)) {
            value = parseFloat(value);
        }

        setTrainingConfig(prev => ({
            ...prev,
            [field]: value
        }));
    };

    const handleSwitchChange = (field) => (event) => {
        setTrainingConfig(prev => ({
            ...prev,
            [field]: event.target.checked
        }));
    };

    const validateConfig = () => {
        const errors = [];
        if (!trainingConfig.modelType) errors.push("Model type is required");
        if (!trainingConfig.optimizer) errors.push("Optimizer is required");
        if (!trainingConfig.loss) errors.push("Loss function is required");
        if (trainingConfig.epochs < 1) errors.push("Epochs must be at least 1");
        if (trainingConfig.batchSize < 1) errors.push("Batch size must be at least 1");
        if (trainingConfig.learningRate <= 0) errors.push("Learning rate must be greater than 0");
        if (trainingConfig.validationSplit < 0 || trainingConfig.validationSplit > 1) {
            errors.push("Validation split must be between 0 and 1");
        }
        return errors;
    };

    const handleStartTraining = async () => {
        try {
            setLoading(true);
            setError(null);
            setSuccess(null);

            // Validate config
            const validationErrors = validateConfig();
            if (validationErrors.length > 0) {
                setError(validationErrors.join(", "));
                return;
            }

            const response = await modelService.startTraining(trainingConfig);
            
            setModelId(response.model_id);
            setSuccess(`Training started successfully! Model ID: ${response.model_id}`);
        } catch (err) {
            setError(err.message || 'Failed to start training');
            console.error('Training error:', err);
        } finally {
            setLoading(false);
        }
    };

    const handleStopTraining = async () => {
        try {
            if (modelId) {
                await modelService.stopTraining(modelId);
                setSuccess('Training stopped successfully');
            }
        } catch (err) {
            setError(err.message || 'Failed to stop training');
            console.error('Error stopping training:', err);
        }
    };

    const renderError = () => {
        if (!error) return null;
        
        // Handle array of errors
        if (Array.isArray(error)) {
            return (
                <Alert severity="error" sx={{ mb: 2 }}>
                    <ul style={{ margin: 0, paddingLeft: 20 }}>
                        {error.map((err, index) => (
                            <li key={index}>{err}</li>
                        ))}
                    </ul>
                </Alert>
            );
        }

        // Handle string error
        return (
            <Alert severity="error" sx={{ mb: 2 }}>
                {error}
            </Alert>
        );
    };

    return (
        <ErrorBoundary>
            <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
                <Paper sx={{ p: 3 }}>
                    <Typography variant="h5" gutterBottom>
                        Model Training Configuration
                    </Typography>

                    {renderError()}

                    {success && (
                        <Alert severity="success" sx={{ mb: 2 }}>
                            {success}
                        </Alert>
                    )}

                    <Grid container spacing={3}>
                        <Grid item xs={12} md={6}>
                            <FormControl fullWidth sx={{ mb: 2 }}>
                                <InputLabel>Model Type</InputLabel>
                                <Select
                                    value={trainingConfig.modelType}
                                    onChange={handleInputChange('modelType')}
                                    label="Model Type"
                                >
                                    <MenuItem value="bert">BERT</MenuItem>
                                    <MenuItem value="gpt">GPT</MenuItem>
                                    <MenuItem value="t5">T5</MenuItem>
                                </Select>
                            </FormControl>

                            <FormControl fullWidth sx={{ mb: 2 }}>
                                <InputLabel>Optimizer</InputLabel>
                                <Select
                                    value={trainingConfig.optimizer}
                                    onChange={handleInputChange('optimizer')}
                                    label="Optimizer"
                                >
                                    <MenuItem value="adam">Adam</MenuItem>
                                    <MenuItem value="sgd">SGD</MenuItem>
                                    <MenuItem value="rmsprop">RMSprop</MenuItem>
                                </Select>
                            </FormControl>

                            <FormControl fullWidth sx={{ mb: 2 }}>
                                <InputLabel>Loss Function</InputLabel>
                                <Select
                                    value={trainingConfig.loss}
                                    onChange={handleInputChange('loss')}
                                    label="Loss Function"
                                >
                                    <MenuItem value="categorical_crossentropy">Categorical Crossentropy</MenuItem>
                                    <MenuItem value="binary_crossentropy">Binary Crossentropy</MenuItem>
                                    <MenuItem value="mse">Mean Squared Error</MenuItem>
                                </Select>
                            </FormControl>
                        </Grid>

                        <Grid item xs={12} md={6}>
                            <TextField
                                fullWidth
                                label="Batch Size"
                                type="number"
                                value={trainingConfig.batchSize}
                                onChange={handleInputChange('batchSize')}
                                sx={{ mb: 2 }}
                            />

                            <TextField
                                fullWidth
                                label="Epochs"
                                type="number"
                                value={trainingConfig.epochs}
                                onChange={handleInputChange('epochs')}
                                sx={{ mb: 2 }}
                            />

                            <TextField
                                fullWidth
                                label="Learning Rate"
                                type="number"
                                value={trainingConfig.learningRate}
                                onChange={handleInputChange('learningRate')}
                                inputProps={{ step: 0.001 }}
                                sx={{ mb: 2 }}
                            />

                            <TextField
                                fullWidth
                                label="Validation Split"
                                type="number"
                                value={trainingConfig.validationSplit}
                                onChange={handleInputChange('validationSplit')}
                                inputProps={{ step: 0.1, min: 0, max: 1 }}
                                sx={{ mb: 2 }}
                            />
                        </Grid>
                    </Grid>

                    {modelId && (
                        <Box sx={{ mt: 2, mb: 2 }}>
                            <Typography variant="subtitle1" gutterBottom>
                                Training Status:
                            </Typography>
                            <Chip
                                label={`Model ID: ${modelId}`}
                                color="primary"
                                sx={{ mr: 1 }}
                            />
                            <Chip
                                label="Training"
                                color="secondary"
                            />
                        </Box>
                    )}

                    <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
                        <Button
                            variant="contained"
                            onClick={handleStartTraining}
                            disabled={loading}
                            sx={{ minWidth: 150 }}
                        >
                            {loading ? (
                                <CircularProgress size={24} color="inherit" />
                            ) : (
                                'Start Training'
                            )}
                        </Button>
                    </Box>
                </Paper>

                {modelId && (
                    <TrainingProgress
                        modelId={modelId}
                        onStopTraining={handleStopTraining}
                    />
                )}
            </Container>
        </ErrorBoundary>
    );
};

export default ModelTraining; 