import React, { useState, useEffect } from 'react';
import {
  Grid,
  Paper,
  Typography,
  Button,
  Box,
  Alert,
  CircularProgress,
  Stepper,
  Step,
  StepLabel,
  Card,
  CardContent,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Switch,
  FormControlLabel
} from '@mui/material';
import { useSelector } from 'react-redux';
import { modelService } from '../../services/modelService';

const deploymentSteps = [
  'Model Validation',
  'Configuration',
  'Resource Allocation',
  'Deployment'
];

function ModelDeployment() {
  const [activeStep, setActiveStep] = useState(0);
  const [deploymentStatus, setDeploymentStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [deploymentConfig, setDeploymentConfig] = useState({
    version: '1.0.0',
    environment: 'staging',
    scalingEnabled: true,
    minInstances: 1,
    maxInstances: 3,
    cpuLimit: '1',
    memoryLimit: '2Gi',
    endpoint: '',
    autoScaling: true
  });

  const modelId = useSelector(state => state.training.modelId);
  const [deployedModels, setDeployedModels] = useState([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    fetchDeployedModels();
  }, []);

  const fetchDeployedModels = async () => {
    try {
      setLoading(true);
      const response = await modelService.getDeployedModels();
      setDeployedModels(response.data);
    } catch (err) {
      setError('Failed to fetch deployed models');
    } finally {
      setLoading(false);
    }
  };

  const handleDeploy = async () => {
    try {
      setDeploymentStatus('deploying');
      setError(null);

      const response = await modelService.deployModel(modelId, deploymentConfig);
      setDeploymentStatus('deployed');
      fetchDeployedModels();
      setActiveStep(deploymentSteps.length);
    } catch (err) {
      setError(err.message || 'Deployment failed');
      setDeploymentStatus('failed');
    }
  };

  const handleNext = () => {
    setActiveStep((prevStep) => prevStep + 1);
    if (activeStep === deploymentSteps.length - 1) {
      handleDeploy();
    }
  };

  const handleBack = () => {
    setActiveStep((prevStep) => prevStep - 1);
  };

  const renderStepContent = (step) => {
    switch (step) {
      case 0:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Model Validation
              </Typography>
              <Box sx={{ mb: 2 }}>
                <Typography>Model ID: {modelId}</Typography>
                <Typography>Status: Ready for deployment</Typography>
              </Box>
            </CardContent>
          </Card>
        );

      case 1:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Deployment Configuration
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Version"
                    value={deploymentConfig.version}
                    onChange={(e) => setDeploymentConfig({
                      ...deploymentConfig,
                      version: e.target.value
                    })}
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth>
                    <InputLabel>Environment</InputLabel>
                    <Select
                      value={deploymentConfig.environment}
                      onChange={(e) => setDeploymentConfig({
                        ...deploymentConfig,
                        environment: e.target.value
                      })}
                    >
                      <MenuItem value="staging">Staging</MenuItem>
                      <MenuItem value="production">Production</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 2:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Resource Allocation
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12}>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={deploymentConfig.autoScaling}
                        onChange={(e) => setDeploymentConfig({
                          ...deploymentConfig,
                          autoScaling: e.target.checked
                        })}
                      />
                    }
                    label="Enable Auto Scaling"
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Min Instances"
                    value={deploymentConfig.minInstances}
                    onChange={(e) => setDeploymentConfig({
                      ...deploymentConfig,
                      minInstances: parseInt(e.target.value)
                    })}
                  />
                </Grid>
                <Grid item xs={6}>
                  <TextField
                    fullWidth
                    type="number"
                    label="Max Instances"
                    value={deploymentConfig.maxInstances}
                    onChange={(e) => setDeploymentConfig({
                      ...deploymentConfig,
                      maxInstances: parseInt(e.target.value)
                    })}
                  />
                </Grid>
              </Grid>
            </CardContent>
          </Card>
        );

      case 3:
        return (
          <Card>
            <CardContent>
              <Typography variant="h6" gutterBottom>
                Deployment Summary
              </Typography>
              <pre>
                {JSON.stringify(deploymentConfig, null, 2)}
              </pre>
            </CardContent>
          </Card>
        );

      default:
        return null;
    }
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Model Deployment
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <Stepper activeStep={activeStep} sx={{ mb: 4 }}>
            {deploymentSteps.map((label) => (
              <Step key={label}>
                <StepLabel>{label}</StepLabel>
              </Step>
            ))}
          </Stepper>

          <Box sx={{ mb: 4 }}>
            {renderStepContent(activeStep)}
          </Box>

          <Box sx={{ display: 'flex', justifyContent: 'flex-end' }}>
            <Button
              disabled={activeStep === 0}
              onClick={handleBack}
              sx={{ mr: 1 }}
            >
              Back
            </Button>
            <Button
              variant="contained"
              onClick={handleNext}
              disabled={deploymentStatus === 'deploying'}
            >
              {activeStep === deploymentSteps.length - 1 ? 'Deploy' : 'Next'}
              {deploymentStatus === 'deploying' && (
                <CircularProgress size={24} sx={{ ml: 1 }} />
              )}
            </Button>
          </Box>
        </Paper>
      </Grid>

      <Grid item xs={12}>
        <Paper sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Deployed Models
          </Typography>
          {loading ? (
            <CircularProgress />
          ) : (
            <Grid container spacing={2}>
              {deployedModels.map((model) => (
                <Grid item xs={12} md={6} key={model.id}>
                  <Card>
                    <CardContent>
                      <Typography variant="h6">{model.name}</Typography>
                      <Typography>Version: {model.version}</Typography>
                      <Typography>Environment: {model.environment}</Typography>
                      <Typography>Status: {model.status}</Typography>
                      <Typography>Endpoint: {model.endpoint}</Typography>
                    </CardContent>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </Paper>
      </Grid>
    </Grid>
  );
}

export default ModelDeployment; 