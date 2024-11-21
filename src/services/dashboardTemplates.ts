interface LayoutItem {
  i: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

interface WidgetConfig {
  id: string;
  type: string;
  config: Record<string, any>;
}

interface DashboardConfig {
  name: string;
  layout: {
    lg: LayoutItem[];
  };
  widgets: WidgetConfig[];
}

export class DashboardTemplate {
  static readonly DEFAULT_TEMPLATES: Record<string, DashboardConfig> = {
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
    }
    // Add more templates here
  };

  static getTemplate(templateId: string): DashboardConfig | undefined {
    return this.DEFAULT_TEMPLATES[templateId];
  }

  static saveCustomTemplate(template: DashboardConfig): void {
    const customTemplates = this.getCustomTemplates();
    customTemplates[template.name] = template;
    localStorage.setItem('customTemplates', JSON.stringify(customTemplates));
  }

  static getCustomTemplates(): Record<string, DashboardConfig> {
    return JSON.parse(localStorage.getItem('customTemplates') || '{}');
  }

  static getAllTemplates(): Record<string, DashboardConfig> {
    return {
      ...this.DEFAULT_TEMPLATES,
      ...this.getCustomTemplates()
    };
  }
} 