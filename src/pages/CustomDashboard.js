import React, { useState } from 'react';
import DashboardLayout from '../components/dashboard/DashboardLayout';
import Heatmap from '../components/visualization/Heatmap';
import ScatterPlot3D from '../components/visualization/ScatterPlot3D';
import InteractiveChart from '../components/monitoring/InteractiveChart';

const defaultLayouts = {
  lg: [
    { i: 'metrics', x: 0, y: 0, w: 8, h: 4 },
    { i: 'heatmap', x: 8, y: 0, w: 4, h: 4 },
    { i: 'scatter3d', x: 0, y: 4, w: 6, h: 4 },
    { i: 'performance', x: 6, y: 4, w: 6, h: 4 }
  ]
};

function CustomDashboard() {
  const [layouts, setLayouts] = useState(defaultLayouts);

  const handleLayoutChange = (layout, layouts) => {
    setLayouts(layouts);
  };

  return (
    <DashboardLayout layouts={layouts} onLayoutChange={handleLayoutChange}>
      <InteractiveChart
        key="metrics"
        data={metricsData}
        title="Model Metrics"
        metrics={['accuracy', 'loss']}
      />
      <Heatmap
        key="heatmap"
        data={heatmapData}
        title="Confusion Matrix"
        options={{
          xAxisLabel: 'Predicted',
          yAxisLabel: 'Actual'
        }}
      />
      <ScatterPlot3D
        key="scatter3d"
        data={scatter3dData}
        title="Model Performance in 3D"
        options={{
          metrics: [
            { value: 'accuracy', label: 'Accuracy' },
            { value: 'latency', label: 'Latency' },
            { value: 'memory', label: 'Memory Usage' }
          ]
        }}
      />
      <InteractiveChart
        key="performance"
        data={performanceData}
        title="Training Performance"
        metrics={['training_loss', 'validation_loss']}
      />
    </DashboardLayout>
  );
}

export default CustomDashboard; 