import React from 'react';
import {
  Dialog,
  DialogTitle,
  DialogContent,
  Grid,
  Card,
  CardContent,
  CardMedia,
  Typography,
  Button
} from '@mui/material';
import DashboardTemplate from '../../services/dashboardTemplates';

function TemplateSelector({ open, onClose, onSelect }) {
  const templates = DashboardTemplate.getAllTemplates();

  return (
    <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
      <DialogTitle>Select Dashboard Template</DialogTitle>
      <DialogContent>
        <Grid container spacing={3} sx={{ mt: 1 }}>
          {Object.entries(templates).map(([id, template]) => (
            <Grid item xs={12} sm={6} key={id}>
              <Card>
                <CardMedia
                  component="img"
                  height="140"
                  image={template.thumbnail || '/default-template-thumb.png'}
                  alt={template.name}
                />
                <CardContent>
                  <Typography gutterBottom variant="h6" component="div">
                    {template.name}
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    {template.description}
                  </Typography>
                  <Button
                    variant="contained"
                    fullWidth
                    sx={{ mt: 2 }}
                    onClick={() => {
                      onSelect(id, template);
                      onClose();
                    }}
                  >
                    Use Template
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </DialogContent>
    </Dialog>
  );
}

export default TemplateSelector; 