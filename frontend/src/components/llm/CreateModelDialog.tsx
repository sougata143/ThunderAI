import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  TextField,
  Button,
  Box,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Grid,
  Typography,
  Divider,
  FormHelperText,
} from '@mui/material';

interface CreateModelDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (model: any) => void;
  error?: string;
}

interface ModelFormData {
  name: string;
  model_type: string;
  version: string;
  description: string;
  parameters: {
    architecture: string;
    num_layers: number;
    hidden_size: number;
    vocab_size: number;
    max_position_embeddings: number;
    num_attention_heads: number;
    intermediate_size: number;
    dropout_rate: number;
  };
  training: {
    batch_size: number;
    learning_rate: number;
    num_epochs: number;
    warmup_steps: number;
    weight_decay: number;
    gradient_clip_val: number;
  };
}

const defaultParameters: ModelFormData['parameters'] = {
  architecture: 'transformer',
  num_layers: 12,
  hidden_size: 768,
  vocab_size: 50000,
  max_position_embeddings: 512,
  num_attention_heads: 12,
  intermediate_size: 3072,
  dropout_rate: 0.1,
};

const defaultTraining: ModelFormData['training'] = {
  batch_size: 32,
  learning_rate: 5e-5,
  num_epochs: 3,
  warmup_steps: 500,
  weight_decay: 0.01,
  gradient_clip_val: 1.0,
};

export const CreateModelDialog: React.FC<CreateModelDialogProps> = ({
  open,
  onClose,
  onSubmit,
  error,
}) => {
  const [model, setModel] = React.useState<ModelFormData>({
    name: '',
    model_type: '',
    version: '',
    description: '',
    parameters: defaultParameters,
    training: defaultTraining,
  });

  const handleInputChange = (
    e: React.ChangeEvent<HTMLInputElement | { name?: string; value: unknown }>
  ) => {
    const { name, value } = e.target;
    if (!name) return;

    if (name.includes('.')) {
      const [section, field] = name.split('.');
      setModel(prev => ({
        ...prev,
        [section]: {
          ...prev[section as keyof Pick<ModelFormData, 'parameters' | 'training'>],
          [field]: value,
        },
      }));
    } else {
      setModel(prev => ({
        ...prev,
        [name]: value,
      }));
    }
  };

  const handleSubmit = () => {
    if (!model.name || !model.model_type || !model.version) {
      return;
    }
    onSubmit(model);
  };

  const handleClose = () => {
    setModel({
      name: '',
      model_type: '',
      version: '',
      description: '',
      parameters: defaultParameters,
      training: defaultTraining,
    });
    onClose();
  };

  return (
    <Dialog open={open} onClose={handleClose} maxWidth="md" fullWidth>
      <DialogTitle>Add New Model</DialogTitle>
      <DialogContent>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3, mt: 2 }}>
          {/* Basic Information */}
          <Box>
            <Typography variant="h6" gutterBottom>
              Basic Information
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="name"
                  label="Name"
                  required
                  error={!model.name}
                  helperText={!model.name ? "Name is required" : ""}
                  value={model.name}
                  onChange={handleInputChange}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth required error={!model.model_type}>
                  <InputLabel>Model Type</InputLabel>
                  <Select
                    name="model_type"
                    value={model.model_type}
                    label="Model Type"
                    onChange={handleInputChange}
                  >
                    <MenuItem value="gpt">GPT</MenuItem>
                    <MenuItem value="bert">BERT</MenuItem>
                    <MenuItem value="t5">T5</MenuItem>
                    <MenuItem value="llama">LLaMA</MenuItem>
                    <MenuItem value="custom">Custom</MenuItem>
                  </Select>
                  <FormHelperText>
                    {!model.model_type ? "Model Type is required" : ""}
                  </FormHelperText>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="version"
                  label="Version"
                  required
                  error={!model.version}
                  helperText={!model.version ? "Version is required" : "e.g., 1.0.0"}
                  value={model.version}
                  onChange={handleInputChange}
                  fullWidth
                />
              </Grid>
              <Grid item xs={12}>
                <TextField
                  name="description"
                  label="Description"
                  multiline
                  rows={2}
                  value={model.description}
                  onChange={handleInputChange}
                  fullWidth
                />
              </Grid>
            </Grid>
          </Box>

          <Divider />

          {/* Model Parameters */}
          <Box>
            <Typography variant="h6" gutterBottom>
              Model Architecture
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <FormControl fullWidth>
                  <InputLabel>Architecture</InputLabel>
                  <Select
                    name="parameters.architecture"
                    value={model.parameters.architecture}
                    label="Architecture"
                    onChange={handleInputChange}
                  >
                    <MenuItem value="transformer">Transformer</MenuItem>
                    <MenuItem value="encoder-decoder">Encoder-Decoder</MenuItem>
                    <MenuItem value="decoder-only">Decoder Only</MenuItem>
                  </Select>
                </FormControl>
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.num_layers"
                  label="Number of Layers"
                  type="number"
                  value={model.parameters.num_layers}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.hidden_size"
                  label="Hidden Size"
                  type="number"
                  value={model.parameters.hidden_size}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 64 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.vocab_size"
                  label="Vocabulary Size"
                  type="number"
                  value={model.parameters.vocab_size}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1000, step: 1000 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.max_position_embeddings"
                  label="Max Position Embeddings"
                  type="number"
                  value={model.parameters.max_position_embeddings}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 128 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.num_attention_heads"
                  label="Number of Attention Heads"
                  type="number"
                  value={model.parameters.num_attention_heads}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.intermediate_size"
                  label="Intermediate Size"
                  type="number"
                  value={model.parameters.intermediate_size}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 256 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="parameters.dropout_rate"
                  label="Dropout Rate"
                  type="number"
                  value={model.parameters.dropout_rate}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 0, max: 1, step: 0.1 }}
                />
              </Grid>
            </Grid>
          </Box>

          <Divider />

          {/* Training Parameters */}
          <Box>
            <Typography variant="h6" gutterBottom>
              Training Configuration
            </Typography>
            <Grid container spacing={2}>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.batch_size"
                  label="Batch Size"
                  type="number"
                  value={model.training.batch_size}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.learning_rate"
                  label="Learning Rate"
                  type="number"
                  value={model.training.learning_rate}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 0, step: 0.00001 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.num_epochs"
                  label="Number of Epochs"
                  type="number"
                  value={model.training.num_epochs}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 1, step: 1 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.warmup_steps"
                  label="Warmup Steps"
                  type="number"
                  value={model.training.warmup_steps}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 0, step: 100 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.weight_decay"
                  label="Weight Decay"
                  type="number"
                  value={model.training.weight_decay}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 0, step: 0.001 }}
                />
              </Grid>
              <Grid item xs={12} sm={6}>
                <TextField
                  name="training.gradient_clip_val"
                  label="Gradient Clip Value"
                  type="number"
                  value={model.training.gradient_clip_val}
                  onChange={handleInputChange}
                  fullWidth
                  inputProps={{ min: 0, step: 0.1 }}
                />
              </Grid>
            </Grid>
          </Box>
        </Box>
      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 2 }}>
        <Button onClick={handleClose}>Cancel</Button>
        <Button
          onClick={handleSubmit}
          variant="contained"
          disabled={!model.name || !model.model_type || !model.version}
        >
          Create
        </Button>
      </DialogActions>
    </Dialog>
  );
};
