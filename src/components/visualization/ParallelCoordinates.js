import React, { useState } from 'react';
import { Paper, Typography, FormControl, Select, MenuItem, Checkbox, FormGroup, FormControlLabel } from '@mui/material';
import { ParallelCoordinates as ParallelCoordinatesChart } from '@nivo/parallel-coordinates';
import { useTheme } from '@mui/material/styles';

function ParallelCoordinates({ data, title, variables = [], options = {} }) {
  const theme = useTheme();
  const [selectedVariables, setSelectedVariables] = useState(variables.map(v => v.id));
  const [brushing, setBrushing] = useState(true);

  const handleVariableToggle = (variableId) => {
    setSelectedVariables(prev => {
      if (prev.includes(variableId)) {
        return prev.filter(id => id !== variableId);
      }
      return [...prev, variableId];
    });
  };

  const filteredVariables = variables.filter(v => selectedVariables.includes(v.id));

  return (
    <Paper elevation={3} sx={{ p: 2 }}>
      <Typography variant="h6" gutterBottom>
        {title}
      </Typography>
      <FormGroup row sx={{ mb: 2 }}>
        {variables.map(variable => (
          <FormControlLabel
            key={variable.id}
            control={
              <Checkbox
                checked={selectedVariables.includes(variable.id)}
                onChange={() => handleVariableToggle(variable.id)}
              />
            }
            label={variable.label}
          />
        ))}
        <FormControlLabel
          control={
            <Checkbox
              checked={brushing}
              onChange={(e) => setBrushing(e.target.checked)}
            />
          }
          label="Enable Brushing"
        />
      </FormGroup>
      <div style={{ height: '400px' }}>
        <ParallelCoordinatesChart
          data={data}
          variables={filteredVariables}
          margin={{ top: 50, right: 60, bottom: 50, left: 60 }}
          theme={{
            textColor: theme.palette.text.primary,
            fontSize: 11,
            axis: {
              domain: {
                line: {
                  stroke: theme.palette.divider,
                  strokeWidth: 1
                }
              },
              ticks: {
                line: {
                  stroke: theme.palette.divider,
                  strokeWidth: 1
                }
              }
            }
          }}
          animate={false}
          strokeWidth={2}
          enableBrushing={brushing}
          brushSize={8}
          {...options}
        />
      </div>
    </Paper>
  );
}

export default ParallelCoordinates; 