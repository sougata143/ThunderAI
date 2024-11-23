import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Box,
  Typography,
} from '@mui/material';

interface TrainingConfig {
  batch_size: number;
  learning_rate: number;
  epochs: number;
  max_length: number;
  model_name: string;
}

interface ModelData {
  name: string;
  description: string;
  architecture: string;
  training_config: TrainingConfig;
  status: string;
  created_by: string;
}

interface CreateModelDialogProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (modelData: ModelData) => void;
}

export const CreateModelDialog: React.FC<CreateModelDialogProps> = ({
  open,
  onClose,
  onSubmit,
}) => {
  const [formData, setFormData] = useState<ModelData>({
    name: '',
    description: '',
    architecture: 'gpt2',
    training_config: {
      batch_size: 32,
      learning_rate: 5e-5,
      epochs: 3,
      max_length: 512,
      model_name: 'gpt2',
    },
    status: 'initialized',
    created_by: 'anonymous',
  });

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value,
    }));
  };

  const handleTrainingConfigChange = (field: keyof TrainingConfig, value: number | string) => {
    setFormData((prev) => ({
      ...prev,
      training_config: {
        ...prev.training_config,
        [field]: value,
      },
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit(formData);
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <form onSubmit={handleSubmit}>
        <DialogTitle>Create New Model</DialogTitle>
        <DialogContent>
          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
            <TextField
              required
              name="name"
              label="Model Name"
              value={formData.name}
              onChange={handleChange}
              fullWidth
            />
            <TextField
              required
              name="description"
              label="Description"
              value={formData.description}
              onChange={handleChange}
              fullWidth
              multiline
              rows={3}
            />
            <FormControl fullWidth required>
              <InputLabel>Architecture</InputLabel>
              <Select
                name="architecture"
                value={formData.architecture}
                label="Architecture"
                onChange={(e) =>
                  setFormData((prev) => ({
                    ...prev,
                    architecture: e.target.value,
                  }))
                }
              >
                <MenuItem value="gpt2">GPT-2</MenuItem>
                <MenuItem value="bert">BERT</MenuItem>
                <MenuItem value="t5">T5</MenuItem>
                <MenuItem value="llama">LLaMA</MenuItem>
              </Select>
            </FormControl>

            <Typography variant="subtitle1" sx={{ mt: 2 }}>
              Training Configuration
            </Typography>

            <TextField
              required
              type="number"
              label="Batch Size"
              value={formData.training_config.batch_size}
              onChange={(e) =>
                handleTrainingConfigChange('batch_size', parseInt(e.target.value))
              }
              fullWidth
              inputProps={{ min: 1 }}
            />
            <TextField
              required
              type="number"
              label="Learning Rate"
              value={formData.training_config.learning_rate}
              onChange={(e) =>
                handleTrainingConfigChange('learning_rate', parseFloat(e.target.value))
              }
              fullWidth
              inputProps={{ step: '0.00001', min: 0 }}
            />
            <TextField
              required
              type="number"
              label="Epochs"
              value={formData.training_config.epochs}
              onChange={(e) =>
                handleTrainingConfigChange('epochs', parseInt(e.target.value))
              }
              fullWidth
              inputProps={{ min: 1 }}
            />
            <TextField
              required
              type="number"
              label="Max Length"
              value={formData.training_config.max_length}
              onChange={(e) =>
                handleTrainingConfigChange('max_length', parseInt(e.target.value))
              }
              fullWidth
              inputProps={{ min: 1 }}
            />
          </Box>
        </DialogContent>
        <DialogActions>
          <Button onClick={onClose}>Cancel</Button>
          <Button type="submit" variant="contained" color="primary">
            Create
          </Button>
        </DialogActions>
      </form>
    </Dialog>
  );
};
