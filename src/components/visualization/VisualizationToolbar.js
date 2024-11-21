import React, { useState } from 'react';
import {
  Box,
  IconButton,
  Tooltip,
  Menu,
  MenuItem,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Snackbar,
  Alert
} from '@mui/material';
import {
  MoreVert as MoreVertIcon,
  Share as ShareIcon,
  Download as DownloadIcon,
  Edit as EditIcon,
  Fullscreen as FullscreenIcon,
  FilterList as FilterListIcon
} from '@mui/icons-material';
import { ExportService } from '../../services/exportService';
import { SharingService } from '../../services/sharingService';

function VisualizationToolbar({ 
  data, 
  title,
  onEdit,
  onFilter,
  exportOptions = ['csv', 'excel', 'image'],
  className 
}) {
  const [anchorEl, setAnchorEl] = useState(null);
  const [shareDialogOpen, setShareDialogOpen] = useState(false);
  const [shareUrl, setShareUrl] = useState('');
  const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });
  const [isFullscreen, setIsFullscreen] = useState(false);

  const handleExport = async (format) => {
    try {
      switch (format) {
        case 'csv':
          await ExportService.exportToCSV(data, title);
          break;
        case 'excel':
          await ExportService.exportToExcel(data, title);
          break;
        case 'image':
          await ExportService.exportToPNG(data, title);
          break;
      }
      showSnackbar('Export successful', 'success');
    } catch (error) {
      showSnackbar('Export failed', 'error');
    }
    setAnchorEl(null);
  };

  const handleShare = async () => {
    try {
      const url = await SharingService.shareVisualization({
        title,
        data,
        config: { /* visualization config */ }
      });
      setShareUrl(url);
      setShareDialogOpen(true);
    } catch (error) {
      showSnackbar('Failed to generate sharing link', 'error');
    }
  };

  const handleFullscreen = () => {
    const element = document.getElementById(`visualization-${title}`);
    if (element) {
      if (!isFullscreen) {
        if (element.requestFullscreen) {
          element.requestFullscreen();
        }
      } else {
        if (document.exitFullscreen) {
          document.exitFullscreen();
        }
      }
      setIsFullscreen(!isFullscreen);
    }
  };

  const showSnackbar = (message, severity) => {
    setSnackbar({ open: true, message, severity });
  };

  return (
    <Box className={className}>
      <Tooltip title="Filter">
        <IconButton onClick={onFilter}>
          <FilterListIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Edit">
        <IconButton onClick={onEdit}>
          <EditIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Share">
        <IconButton onClick={handleShare}>
          <ShareIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Export">
        <IconButton onClick={(e) => setAnchorEl(e.currentTarget)}>
          <DownloadIcon />
        </IconButton>
      </Tooltip>
      <Tooltip title="Fullscreen">
        <IconButton onClick={handleFullscreen}>
          <FullscreenIcon />
        </IconButton>
      </Tooltip>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={() => setAnchorEl(null)}
      >
        {exportOptions.includes('csv') && (
          <MenuItem onClick={() => handleExport('csv')}>Export as CSV</MenuItem>
        )}
        {exportOptions.includes('excel') && (
          <MenuItem onClick={() => handleExport('excel')}>Export as Excel</MenuItem>
        )}
        {exportOptions.includes('image') && (
          <MenuItem onClick={() => handleExport('image')}>Export as Image</MenuItem>
        )}
      </Menu>

      <Dialog open={shareDialogOpen} onClose={() => setShareDialogOpen(false)}>
        <DialogTitle>Share Visualization</DialogTitle>
        <DialogContent>
          <TextField
            fullWidth
            value={shareUrl}
            label="Sharing URL"
            variant="outlined"
            margin="normal"
            InputProps={{
              readOnly: true,
            }}
          />
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setShareDialogOpen(false)}>Close</Button>
          <Button
            variant="contained"
            onClick={() => {
              navigator.clipboard.writeText(shareUrl);
              showSnackbar('URL copied to clipboard', 'success');
            }}
          >
            Copy URL
          </Button>
        </DialogActions>
      </Dialog>

      <Snackbar
        open={snackbar.open}
        autoHideDuration={6000}
        onClose={() => setSnackbar({ ...snackbar, open: false })}
      >
        <Alert severity={snackbar.severity} onClose={() => setSnackbar({ ...snackbar, open: false })}>
          {snackbar.message}
        </Alert>
      </Snackbar>
    </Box>
  );
}

export default VisualizationToolbar; 