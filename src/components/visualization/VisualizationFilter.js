import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  TextField,
  Box,
  Chip,
  Slider,
  Typography
} from '@mui/material';

function VisualizationFilter({ open, onClose, filters, onApplyFilters }) {
  const [activeFilters, setActiveFilters] = useState(filters);

  const handleFilterChange = (filterId, changes) => {
    setActiveFilters(prev => ({
      ...prev,
      [filterId]: {
        ...prev[filterId],
        ...changes
      }
    }));
  };

  const renderFilterControl = (filter) => {
    switch (filter.type) {
      case 'range':
        return (
          <Box sx={{ width: '100%', mt: 2 }}>
            <Typography gutterBottom>{filter.label}</Typography>
            <Slider
              value={[filter.min, filter.max]}
              onChange={(_, newValue) => 
                handleFilterChange(filter.id, { min: newValue[0], max: newValue[1] })
              }
              valueLabelDisplay="auto"
              min={filter.absoluteMin}
              max={filter.absoluteMax}
            />
          </Box>
        );
      case 'select':
        return (
          <FormControl fullWidth sx={{ mt: 2 }}>
            <InputLabel>{filter.label}</InputLabel>
            <Select
              multiple
              value={filter.selected}
              onChange={(e) => handleFilterChange(filter.id, { selected: e.target.value })}
              renderValue={(selected) => (
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                  {selected.map((value) => (
                    <Chip key={value} label={value} />
                  ))}
                </Box>
              )}
            >
              {filter.options.map((option) => (
                <MenuItem key={option} value={option}>
                  {option}
                </MenuItem>
              ))}
            </Select>
          </FormControl>
        );
      case 'search':
        return (
          <TextField
            fullWidth
            label={filter.label}
            value={filter.value}
            onChange={(e) => handleFilterChange(filter.id, { value: e.target.value })}
            sx={{ mt: 2 }}
          />
        );
      default:
        return null;
    }
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="sm" fullWidth>
      <DialogTitle>Filter Visualization</DialogTitle>
      <DialogContent>
        {Object.entries(activeFilters).map(([filterId, filter]) => (
          <Box key={filterId}>
            {renderFilterControl({ id: filterId, ...filter })}
          </Box>
        ))}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button
          variant="contained"
          onClick={() => {
            onApplyFilters(activeFilters);
            onClose();
          }}
        >
          Apply Filters
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default VisualizationFilter; 