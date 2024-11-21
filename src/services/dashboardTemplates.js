class DashboardTemplate {
  static readonly DEFAULT_TEMPLATES = {
    monitoring: {
      name: 'Monitoring Dashboard',
      layout: {
        lg: [
          { i: 'realtime-metrics', x: 0, y: 0, w: 12, h: 4 },
          { i: 'model-performance', x: 0, y: 4, w: 6, h: 4 },
          { i: 'system-resources', x: 6, y: 4, w: 6, h: 4 },
          { i: 'prediction-distribution', x: 0, y: 8, w: 12, h: 4 }
        ]
      },
      widgets: [
        {
          id: 'realtime-metrics',
          type: 'RealTimeMetrics',
          config: { refreshInterval: 5000 }
        },
        {
          id: 'model-performance',
          type: 'PerformanceAnalysis',
          config: { metric: 'accuracy' }
        },
        {
          id: 'system-resources',
          type: 'SystemMetrics',
          config: { showGPU: true }
        },
        {
          id: 'prediction-distribution',
          type: 'PredictionAnalysis',
          config: { chartType: 'bar' }
        }
      ]
    },
    experimentation: {
      name: 'Experimentation Dashboard',
      layout: {
        lg: [
          { i: 'experiment-overview', x: 0, y: 0, w: 12, h: 3 },
          { i: 'parallel-coords', x: 0, y: 3, w: 6, h: 5 },
          { i: 'scatter-3d', x: 6, y: 3, w: 6, h: 5 },
          { i: 'training-history', x: 0, y: 8, w: 12, h: 4 }
        ]
      },
      widgets: [
        {
          id: 'experiment-overview',
          type: 'ExperimentOverview',
          config: { showMetrics: true }
        },
        {
          id: 'parallel-coords',
          type: 'ParallelCoordinates',
          config: { enableBrushing: true }
        },
        {
          id: 'scatter-3d',
          type: 'ScatterPlot3D',
          config: { interactive: true }
        },
        {
          id: 'training-history',
          type: 'TrainingHistory',
          config: { metrics: ['loss', 'accuracy'] }
        }
      ]
    }
  };

  static getTemplate(templateId) {
    return this.DEFAULT_TEMPLATES[templateId];
  }

  static saveCustomTemplate(template) {
    const customTemplates = JSON.parse(localStorage.getItem('customTemplates') || '{}');
    customTemplates[template.id] = template;
    localStorage.setItem('customTemplates', JSON.stringify(customTemplates));
  }

  static getCustomTemplates() {
    return JSON.parse(localStorage.getItem('customTemplates') || '{}');
  }

  static getAllTemplates() {
    return {
      ...this.DEFAULT_TEMPLATES,
      ...this.getCustomTemplates()
    };
  }
}

export default DashboardTemplate; 