import React, { useState, useEffect } from 'react';
import {
  Paper,
  IconButton,
  Tooltip,
  Popover,
  TextField,
  Button,
  List,
  ListItem,
  ListItemText,
  ListItemSecondaryAction,
  Typography
} from '@mui/material';
import {
  Comment as CommentIcon,
  Delete as DeleteIcon,
  Edit as EditIcon
} from '@mui/icons-material';
import CollaborationService from '../../services/collaborationService';

function Annotations({ widgetId }) {
  const [annotations, setAnnotations] = useState([]);
  const [anchorEl, setAnchorEl] = useState(null);
  const [newAnnotation, setNewAnnotation] = useState('');
  const [editingAnnotation, setEditingAnnotation] = useState(null);

  useEffect(() => {
    CollaborationService.subscribe('annotation_added', handleAnnotationAdded);
    CollaborationService.subscribe('annotation_updated', handleAnnotationUpdated);
    CollaborationService.subscribe('annotation_deleted', handleAnnotationDeleted);

    return () => {
      CollaborationService.unsubscribe('annotation_added', handleAnnotationAdded);
      CollaborationService.unsubscribe('annotation_updated', handleAnnotationUpdated);
      CollaborationService.unsubscribe('annotation_deleted', handleAnnotationDeleted);
    };
  }, []);

  const handleAnnotationAdded = (annotation) => {
    if (annotation.widgetId === widgetId) {
      setAnnotations(prev => [...prev, annotation]);
    }
  };

  const handleAnnotationUpdated = (updatedAnnotation) => {
    if (updatedAnnotation.widgetId === widgetId) {
      setAnnotations(prev =>
        prev.map(ann =>
          ann.id === updatedAnnotation.id ? updatedAnnotation : ann
        )
      );
    }
  };

  const handleAnnotationDeleted = (annotationId) => {
    setAnnotations(prev => prev.filter(ann => ann.id !== annotationId));
  };

  const handleAddAnnotation = () => {
    if (newAnnotation.trim()) {
      CollaborationService.addAnnotation({
        widgetId,
        content: newAnnotation,
        timestamp: new Date().toISOString()
      });
      setNewAnnotation('');
      setAnchorEl(null);
    }
  };

  const handleEditAnnotation = (annotation) => {
    if (editingAnnotation?.content !== annotation.content) {
      CollaborationService.updateAnnotation({
        ...annotation,
        content: editingAnnotation.content
      });
    }
    setEditingAnnotation(null);
  };

  const handleDeleteAnnotation = (annotationId) => {
    CollaborationService.deleteAnnotation(annotationId);
  };

  return (
    <>
      <Tooltip title="Annotations">
        <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
          <CommentIcon />
        </IconButton>
      </Tooltip>

      <Popover
        open={Boolean(anchorEl)}
        anchorEl={anchorEl}
        onClose={() => setAnchorEl(null)}
        anchorOrigin={{
          vertical: 'bottom',
          horizontal: 'right',
        }}
        transformOrigin={{
          vertical: 'top',
          horizontal: 'right',
        }}
      >
        <Paper sx={{ width: 320, p: 2 }}>
          <Typography variant="h6" gutterBottom>
            Annotations
          </Typography>

          <List>
            {annotations.map((annotation) => (
              <ListItem key={annotation.id}>
                {editingAnnotation?.id === annotation.id ? (
                  <TextField
                    fullWidth
                    value={editingAnnotation.content}
                    onChange={(e) =>
                      setEditingAnnotation({
                        ...editingAnnotation,
                        content: e.target.value
                      })
                    }
                    onBlur={() => handleEditAnnotation(annotation)}
                    onKeyPress={(e) => e.key === 'Enter' && handleEditAnnotation(annotation)}
                    autoFocus
                  />
                ) : (
                  <>
                    <ListItemText
                      primary={annotation.content}
                      secondary={`${annotation.user.name} - ${new Date(annotation.timestamp).toLocaleString()}`}
                    />
                    <ListItemSecondaryAction>
                      <IconButton
                        edge="end"
                        onClick={() => setEditingAnnotation(annotation)}
                      >
                        <EditIcon />
                      </IconButton>
                      <IconButton
                        edge="end"
                        onClick={() => handleDeleteAnnotation(annotation.id)}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </ListItemSecondaryAction>
                  </>
                )}
              </ListItem>
            ))}
          </List>

          <TextField
            fullWidth
            placeholder="Add annotation..."
            value={newAnnotation}
            onChange={(e) => setNewAnnotation(e.target.value)}
            onKeyPress={(e) => e.key === 'Enter' && handleAddAnnotation()}
            sx={{ mt: 2 }}
          />
          <Button
            fullWidth
            variant="contained"
            onClick={handleAddAnnotation}
            sx={{ mt: 1 }}
          >
            Add Annotation
          </Button>
        </Paper>
      </Popover>
    </>
  );
}

export default Annotations; 