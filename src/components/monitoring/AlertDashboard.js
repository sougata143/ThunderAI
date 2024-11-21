import React, { useState, useEffect } from 'react';
import {
  Paper,
  Typography,
  Grid,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  Box,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel
} from '@mui/material';
import {
  Warning as WarningIcon,
  Error as ErrorIcon,
  Info as InfoIcon,
  Add as AddIcon,
  Delete as DeleteIcon
} from '@mui/icons-material';
import AlertService from '../../services/alertService';

const severityIcons = {
  error: <ErrorIcon color="error" />,
  warning: <WarningIcon color="warning" />,
  info: <InfoIcon color="info" />
};

function AlertDashboard() {
  const [activeAlerts, setActiveAlerts] = useState([]);
  const [alertHistory, setAlertHistory] = useState([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [newRule, setNewRule] = useState({
    id: '',
    metric: '',
    condition: 'threshold',
    threshold: 0,
    severity: 'warning',
    message: '',
    cooldown: 300000
  });

  useEffect(() => {
    const subscription = AlertService.subscribe(({ type, alert }) => {
      if (type === 'triggered') {
        setActiveAlerts(prev => [...prev, alert]);
      } else if (type === 'resolved') {
        setActiveAlerts(prev => prev.filter(a => a.id !== alert.id));
      }
      setAlertHistory(AlertService.getAlertHistory());
    });

    return () => subscription.unsubscribe();
  }, []);

  const handleAddRule = () => {
    AlertService.addRule(newRule.id, {
      condition: (value, threshold) => {
        switch (newRule.condition) {
          case 'threshold':
            return value > threshold;
          case 'below':
            return value < threshold;
          default:
            return false;
        }
      },
      threshold: newRule.threshold,
      severity: newRule.severity,
      message: newRule.message,
      cooldown: newRule.cooldown
    });
    setDialogOpen(false);
    setNewRule({
      id: '',
      metric: '',
      condition: 'threshold',
      threshold: 0,
      severity: 'warning',
      message: '',
      cooldown: 300000
    });
  };

  return (
    <Grid container spacing={3}>
      <Grid item xs={12}>
        <Box display="flex" justifyContent="space-between" alignItems="center">
          <Typography variant="h5">Alerts</Typography>
          <IconButton onClick={() => setDialogOpen(true)}>
            <AddIcon />
          </IconButton>
        </Box>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Active Alerts</Typography>
          <List>
            {activeAlerts.map(alert => (
              <ListItem key={alert.id}>
                <ListItemIcon>
                  {severityIcons[alert.severity]}
                </ListItemIcon>
                <ListItemText
                  primary={alert.message}
                  secondary={new Date(alert.timestamp).toLocaleString()}
                />
                <Chip
                  label={alert.severity}
                  color={alert.severity === 'error' ? 'error' : 'warning'}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Grid>

      <Grid item xs={12} md={6}>
        <Paper elevation={3} sx={{ p: 2 }}>
          <Typography variant="h6" gutterBottom>Alert History</Typography>
          <List>
            {alertHistory.map(alert => (
              <ListItem key={alert.id}>
                <ListItemIcon>
                  {severityIcons[alert.severity]}
                </ListItemIcon>
                <ListItemText
                  primary={alert.message}
                  secondary={`${new Date(alert.timestamp).toLocaleString()} - ${alert.status}`}
                />
              </ListItem>
            ))}
          </List>
        </Paper>
      </Grid>

      <Dialog open={dialogOpen} onClose={() => setDialogOpen(false)}>
        <DialogTitle>Add Alert Rule</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            label="Rule ID"
            value={newRule.id}
            onChange={(e) => setNewRule(prev => ({ ...prev, id: e.target.value }))}
            margin="normal"
          />
          <TextField
            fullWidth
            label="Metric"
            value={newRule.metric}
            onChange={(e) => setNewRule(prev => ({ ...prev, metric: e.target.value }))}
            margin="normal"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Condition</InputLabel>
            <Select
              value={newRule.condition}
              onChange={(e) => setNewRule(prev => ({ ...prev, condition: e.target.value }))}
            >
              <MenuItem value="threshold">Above Threshold</MenuItem>
              <MenuItem value="below">Below Threshold</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            type="number"
            label="Threshold"
            value={newRule.threshold}
            onChange={(e) => setNewRule(prev => ({ ...prev, threshold: parseFloat(e.target.value) }))}
            margin="normal"
          />
          <FormControl fullWidth margin="normal">
            <InputLabel>Severity</InputLabel>
            <Select
              value={newRule.severity}
              onChange={(e) => setNewRule(prev => ({ ...prev, severity: e.target.value }))}
            >
              <MenuItem value="info">Info</MenuItem>
              <MenuItem value="warning">Warning</MenuItem>
              <MenuItem value="error">Error</MenuItem>
            </Select>
          </FormControl>
          <TextField
            fullWidth
            label="Message"
            value={newRule.message}
            onChange={(e) => setNewRule(prev => ({ ...prev, message: e.target.value }))}
            margin="normal"
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setDialogOpen(false)}>Cancel</Button>
          <Button onClick={handleAddRule} variant="contained" color="primary">
            Add Rule
          </Button>
        </DialogActions>
      </Dialog>
    </Grid>
  );
}

export default AlertDashboard; 