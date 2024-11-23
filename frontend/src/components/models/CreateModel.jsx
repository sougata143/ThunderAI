import React, { useState } from 'react';
import {
  Container,
  Paper,
  Typography,
  Stepper,
  Step,
  StepLabel,
  Box,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
} from '@mui/material';
import { useNavigate } from 'react-router-dom';

const steps = ['Basic Information', 'Model Configuration', 'Data Configuration'];

const modelTypes = [
  { value: 'classification', label: 'Classification' },
  { value: 'regression', label: 'Regression' },
  { value: 'nlp', label: 'Natural Language Processing' },
  { value: 'vision', label: 'Computer Vision' },
];

const pretrainedModels = {
  classification: [
    'ResNet50',
    'VGG16',
    'DenseNet121',
  ],
  regression: [
    'LinearRegression',
    'RandomForest',
    'XGBoost',
  ],
  nlp: [
    'BERT-base',
    'RoBERTa',
    'GPT-2-small',
  ],
  vision: [
    'YOLOv5',
    'EfficientNet',
    'MobileNetV3',
  ],
};

const CreateModel = () => {
  const navigate = useNavigate();
  const [activeStep, setActiveStep] = useState(0);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    modelType: '',
    framework: '',
    pretrainedModel: '',
    batchSize: 32,
    learningRate: 0.001,
    epochs: 10,
    datasetPath: '',
    validationSplit: 0.2,
  });

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleSubmit = async () => {
    try {
      // Implement API call to create model
      navigate('/dashboard');
    } catch (error) {
      console.error('Failed to create model:', error);
    }
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Model Name"
                name="name"
                value={formData.name}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Description"
                name="description"
                value={formData.description}
                onChange={handleChange}
                multiline
                rows={4}
              />
            </Grid>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Model Type</InputLabel>
                <Select
                  name="modelType"
                  value={formData.modelType}
                  onChange={handleChange}
                  required
                >
                  {modelTypes.map((type) => (
                    <MenuItem key={type.value} value={type.value}>
                      {type.label}
                    </MenuItem>
                  ))}
                </Select>
              </FormControl>
            </Grid>
          </Grid>
        );

      case 1:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <FormControl fullWidth>
                <InputLabel>Pre-trained Model</InputLabel>
                <Select
                  name="pretrainedModel"
                  value={formData.pretrainedModel}
                  onChange={handleChange}
                  required
                >
                  {formData.modelType &&
                    pretrainedModels[formData.modelType].map((model) => (
                      <MenuItem key={model} value={model}>
                        {model}
                      </MenuItem>
                    ))}
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Batch Size"
                name="batchSize"
                value={formData.batchSize}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Learning Rate"
                name="learningRate"
                value={formData.learningRate}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12} md={4}>
              <TextField
                fullWidth
                type="number"
                label="Epochs"
                name="epochs"
                value={formData.epochs}
                onChange={handleChange}
                required
              />
            </Grid>
          </Grid>
        );

      case 2:
        return (
          <Grid container spacing={3}>
            <Grid item xs={12}>
              <TextField
                fullWidth
                label="Dataset Path"
                name="datasetPath"
                value={formData.datasetPath}
                onChange={handleChange}
                required
              />
            </Grid>
            <Grid item xs={12}>
              <TextField
                fullWidth
                type="number"
                label="Validation Split"
                name="validationSplit"
                value={formData.validationSplit}
                onChange={handleChange}
                required
              />
            </Grid>
          </Grid>
        );

      default:
        return null;
    }
  };

  return (
    <Container maxWidth="md" sx={{ mt: 4, mb: 4 }}>
      <Paper sx={{ p: 4 }}>
        <Typography variant="h4" gutterBottom>
          Create New Model
        </Typography>

        <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
          {steps.map((label) => (
            <Step key={label}>
              <StepLabel>{label}</StepLabel>
            </Step>
          ))}
        </Stepper>

        <form>
          {renderStepContent(activeStep)}

          <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 3 }}>
            {activeStep !== 0 && (
              <Button onClick={handleBack} sx={{ mr: 1 }}>
                Back
              </Button>
            )}
            {activeStep === steps.length - 1 ? (
              <Button variant="contained" color="primary" onClick={handleSubmit}>
                Create Model
              </Button>
            ) : (
              <Button variant="contained" color="primary" onClick={handleNext}>
                Next
              </Button>
            )}
          </Box>
        </form>
      </Paper>
    </Container>
  );
};

export default CreateModel;
