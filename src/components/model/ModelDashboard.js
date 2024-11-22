import React, { useState } from 'react';
import { Grid, Paper, Tabs, Tab, Box } from '@mui/material';
import ModelTraining from './ModelTraining';
import ModelEvaluation from './ModelEvaluation';
import ModelMonitoring from './ModelMonitoring';
import ModelDeployment from './ModelDeployment';

function TabPanel({ children, value, index }) {
  return (
    <div hidden={value !== index}>
      {value === index && <Box p={3}>{children}</Box>}
    </div>
  );
}

function ModelDashboard() {
  const [activeTab, setActiveTab] = useState(0);

  const handleTabChange = (event, newValue) => {
    setActiveTab(newValue);
  };

  return (
    <div>
      <Paper>
        <Tabs value={activeTab} onChange={handleTabChange} centered>
          <Tab label="Training" />
          <Tab label="Evaluation" />
          <Tab label="Monitoring" />
          <Tab label="Deployment" />
        </Tabs>
      </Paper>

      <TabPanel value={activeTab} index={0}>
        <ModelTraining />
      </TabPanel>
      <TabPanel value={activeTab} index={1}>
        <ModelEvaluation />
      </TabPanel>
      <TabPanel value={activeTab} index={2}>
        <ModelMonitoring />
      </TabPanel>
      <TabPanel value={activeTab} index={3}>
        <ModelDeployment />
      </TabPanel>
    </div>
  );
}

export default ModelDashboard; 