import React from 'react';
import {
  Card,
  CardContent,
  CardActions,
  Typography,
  Button,
  Box,
  Chip,
  LinearProgress,
  IconButton,
  Tooltip,
} from '@mui/material';
import {
  PlayArrow as PlayIcon,
  Stop as StopIcon,
  CloudUpload as DeployIcon,
  Assessment as EvaluateIcon,
  Download as DownloadIcon,
} from '@mui/icons-material';
import { useNavigate } from 'react-router-dom';

const statusColors = {
  training: 'warning',
  completed: 'success',
  failed: 'error',
  deployed: 'info',
  stopped: 'default',
};

const ModelCard = ({ model }) => {
  const navigate = useNavigate();

  const handleAction = (action) => {
    switch (action) {
      case 'train':
        navigate(`/model/${model.id}/train`);
        break;
      case 'stop':
        // Implement stop training logic
        break;
      case 'evaluate':
        navigate(`/model/${model.id}/evaluate`);
        break;
      case 'deploy':
        navigate(`/model/${model.id}/deploy`);
        break;
      case 'download':
        // Implement model download logic
        break;
      default:
        break;
    }
  };

  return (
    <Card sx={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      <CardContent sx={{ flexGrow: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="center" mb={2}>
          <Typography variant="h6" component="h2" noWrap>
            {model.name}
          </Typography>
          <Chip
            label={model.status}
            color={statusColors[model.status]}
            size="small"
          />
        </Box>

        <Typography color="textSecondary" gutterBottom>
          {model.type} • {model.framework}
        </Typography>

        {model.status === 'training' && (
          <Box mt={2}>
            <Typography variant="body2" color="textSecondary">
              Training Progress: {model.progress}%
            </Typography>
            <LinearProgress
              variant="determinate"
              value={model.progress}
              sx={{ mt: 1 }}
            />
          </Box>
        )}

        <Box mt={2}>
          <Typography variant="body2">
            Created: {new Date(model.createdAt).toLocaleDateString()}
          </Typography>
          {model.lastTrained && (
            <Typography variant="body2">
              Last Trained: {new Date(model.lastTrained).toLocaleDateString()}
            </Typography>
          )}
        </Box>
      </CardContent>

      <CardActions sx={{ justifyContent: 'flex-end', p: 2 }}>
        {model.status === 'training' ? (
          <Tooltip title="Stop Training">
            <IconButton
              color="error"
              onClick={() => handleAction('stop')}
              size="small"
            >
              <StopIcon />
            </IconButton>
          </Tooltip>
        ) : (
          <Tooltip title="Start Training">
            <IconButton
              color="primary"
              onClick={() => handleAction('train')}
              size="small"
            >
              <PlayIcon />
            </IconButton>
          </Tooltip>
        )}

        <Tooltip title="Evaluate Model">
          <IconButton
            color="primary"
            onClick={() => handleAction('evaluate')}
            size="small"
            disabled={model.status === 'training'}
          >
            <EvaluateIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Deploy Model">
          <IconButton
            color="primary"
            onClick={() => handleAction('deploy')}
            size="small"
            disabled={model.status === 'training' || !model.lastTrained}
          >
            <DeployIcon />
          </IconButton>
        </Tooltip>

        <Tooltip title="Download Model">
          <IconButton
            color="primary"
            onClick={() => handleAction('download')}
            size="small"
            disabled={model.status === 'training' || !model.lastTrained}
          >
            <DownloadIcon />
          </IconButton>
        </Tooltip>
      </CardActions>
    </Card>
  );
};

export default ModelCard;
