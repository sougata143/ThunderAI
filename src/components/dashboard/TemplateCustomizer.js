import React, { useState } from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Grid,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Chip,
  Box
} from '@mui/material';
import { DragDropContext, Droppable, Draggable } from 'react-beautiful-dnd';

function TemplateCustomizer({ open, onClose, template, onSave }) {
  const [customTemplate, setCustomTemplate] = useState(template);
  const [templateName, setTemplateName] = useState(template?.name || '');

  const handleWidgetConfigChange = (widgetId, config) => {
    setCustomTemplate(prev => ({
      ...prev,
      widgets: prev.widgets.map(w =>
        w.id === widgetId ? { ...w, config: { ...w.config, ...config } } : w
      )
    }));
  };

  const handleDragEnd = (result) => {
    if (!result.destination) return;

    const widgets = Array.from(customTemplate.widgets);
    const [reorderedWidget] = widgets.splice(result.source.index, 1);
    widgets.splice(result.destination.index, 0, reorderedWidget);

    setCustomTemplate(prev => ({
      ...prev,
      widgets
    }));
  };

  const handleSave = () => {
    onSave({
      ...customTemplate,
      name: templateName
    });
    onClose();
  };

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Customize Dashboard Template</DialogTitle>
      <DialogContent>
        <Grid container spacing={3} sx={{ mt: 1 }}>
          <Grid item xs={12}>
            <TextField
              fullWidth
              label="Template Name"
              value={templateName}
              onChange={(e) => setTemplateName(e.target.value)}
            />
          </Grid>
          <Grid item xs={12}>
            <DragDropContext onDragEnd={handleDragEnd}>
              <Droppable droppableId="widgets">
                {(provided) => (
                  <Box
                    {...provided.droppableProps}
                    ref={provided.innerRef}
                    sx={{ minHeight: 200 }}
                  >
                    {customTemplate.widgets.map((widget, index) => (
                      <Draggable
                        key={widget.id}
                        draggableId={widget.id}
                        index={index}
                      >
                        {(provided) => (
                          <Box
                            ref={provided.innerRef}
                            {...provided.draggableProps}
                            {...provided.dragHandleProps}
                            sx={{ mb: 2 }}
                          >
                            <Grid container spacing={2} alignItems="center">
                              <Grid item xs={4}>
                                <Chip label={widget.type} />
                              </Grid>
                              <Grid item xs={8}>
                                <FormControl fullWidth size="small">
                                  <InputLabel>Widget Configuration</InputLabel>
                                  <Select
                                    value={widget.config.type || 'default'}
                                    onChange={(e) =>
                                      handleWidgetConfigChange(widget.id, {
                                        type: e.target.value
                                      })
                                    }
                                  >
                                    <MenuItem value="default">Default</MenuItem>
                                    <MenuItem value="compact">Compact</MenuItem>
                                    <MenuItem value="detailed">Detailed</MenuItem>
                                  </Select>
                                </FormControl>
                              </Grid>
                            </Grid>
                          </Box>
                        )}
                      </Draggable>
                    ))}
                    {provided.placeholder}
                  </Box>
                )}
              </Droppable>
            </DragDropContext>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose}>Cancel</Button>
        <Button onClick={handleSave} variant="contained">
          Save Template
        </Button>
      </DialogActions>
    </Dialog>
  );
}

export default TemplateCustomizer; 