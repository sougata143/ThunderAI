import React, { useState, useEffect } from 'react';
import { Responsive, WidthProvider } from 'react-grid-layout';
import { Paper, IconButton, Menu, MenuItem } from '@mui/material';
import MoreVertIcon from '@mui/icons-material/MoreVert';
import 'react-grid-layout/css/styles.css';
import 'react-resizable/css/styles.css';
import CollaborationService from '../../services/collaborationService';
import Collaborators from '../collaboration/Collaborators';
import CollaborationCursor from '../collaboration/CollaborationCursor';
import RealtimeMetricsDashboard from '../monitoring/RealtimeMetricsDashboard';

const ResponsiveGridLayout = WidthProvider(Responsive);

function DashboardLayout({ layouts, children, dashboardId }) {
  const [menuAnchor, setMenuAnchor] = useState(null);
  const [activeWidget, setActiveWidget] = useState(null);
  const [collaborators, setCollaborators] = useState([]);
  const [cursorPositions, setCursorPositions] = useState({});

  useEffect(() => {
    const userId = localStorage.getItem('userId'); // Get from auth context
    const userName = localStorage.getItem('userName');

    CollaborationService.connect(dashboardId, userId, userName);

    CollaborationService.subscribe('collaborators_changed', (users) => {
      setCollaborators(users);
    });

    CollaborationService.subscribe('cursor_moved', ({ userId, position }) => {
      setCursorPositions(prev => ({
        ...prev,
        [userId]: position
      }));
    });

    CollaborationService.subscribe('widget_updated', handleWidgetUpdate);
    CollaborationService.subscribe('layout_changed', handleLayoutChange);

    return () => {
      CollaborationService.disconnect();
    };
  }, [dashboardId]);

  const handleMenuOpen = (event, widgetId) => {
    setMenuAnchor(event.currentTarget);
    setActiveWidget(widgetId);
  };

  const handleMenuClose = () => {
    setMenuAnchor(null);
    setActiveWidget(null);
  };

  const handleMouseMove = (e) => {
    const position = {
      x: e.clientX,
      y: e.clientY
    };
    CollaborationService.updateCursorPosition(position);
  };

  const renderWidget = (child, layoutItem) => (
    <Paper
      key={layoutItem.i}
      elevation={3}
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
        position: 'relative'
      }}
    >
      <div style={{ position: 'absolute', right: 8, top: 8, zIndex: 1 }}>
        <IconButton
          size="small"
          onClick={(e) => handleMenuOpen(e, layoutItem.i)}
        >
          <MoreVertIcon />
        </IconButton>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        {child}
      </div>
    </Paper>
  );

  return (
    <div onMouseMove={handleMouseMove}>
      <ResponsiveGridLayout
        className="layout"
        layouts={layouts}
        breakpoints={{ lg: 1200, md: 996, sm: 768, xs: 480, xxs: 0 }}
        cols={{ lg: 12, md: 10, sm: 6, xs: 4, xxs: 2 }}
        rowHeight={100}
        margin={[16, 16]}
      >
        {React.Children.map(children, (child, index) =>
          renderWidget(child, layouts.lg[index])
        )}
        <div key="realtime-metrics" data-grid={{ x: 0, y: 0, w: 12, h: 8 }}>
          <RealtimeMetricsDashboard />
        </div>
      </ResponsiveGridLayout>

      <Menu
        anchorEl={menuAnchor}
        open={Boolean(menuAnchor)}
        onClose={handleMenuClose}
      >
        <MenuItem onClick={handleMenuClose}>Maximize</MenuItem>
        <MenuItem onClick={handleMenuClose}>Settings</MenuItem>
        <MenuItem onClick={handleMenuClose}>Remove</MenuItem>
      </Menu>

      {/* Render collaborator cursors */}
      {Object.entries(cursorPositions).map(([userId, position]) => {
        const user = collaborators.find(u => u.id === userId);
        if (user) {
          return (
            <CollaborationCursor
              key={userId}
              position={position}
              user={user}
            />
          );
        }
        return null;
      })}

      {/* Show active collaborators */}
      <Collaborators users={collaborators} />
    </div>
  );
}

export default DashboardLayout; 