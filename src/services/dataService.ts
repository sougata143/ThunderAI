interface FilterConfig {
  enabled: boolean;
  type: 'range' | 'categorical' | 'search';
  min?: number;
  max?: number;
  values?: any[];
  value?: string;
}

interface AggregationConfig {
  groupBy: string | ((item: any) => string);
  metrics: Array<{
    name: string;
    type: 'sum' | 'average' | 'count' | 'custom';
    field?: string;
    fn?: (items: any[]) => any;
  }>;
}

interface TransformationConfig {
  type: 'normalize' | 'pivot' | 'timeseries';
  fields?: string[];
  config?: any;
}

export class DataService {
  static filterData(data: any[], filters: Record<string, FilterConfig>): any[] {
    return data.filter(item => {
      return Object.entries(filters).every(([key, filter]) => {
        if (!filter.enabled) return true;
        
        const value = item[key];
        switch (filter.type) {
          case 'range':
            return value >= filter.min && value <= filter.max;
          case 'categorical':
            return filter.values?.includes(value);
          case 'search':
            return value.toLowerCase().includes(filter.value?.toLowerCase() || '');
          default:
            return true;
        }
      });
    });
  }

  static aggregateData(data: any[], aggregation: AggregationConfig): any[] {
    const { groupBy, metrics } = aggregation;
    
    const groups = data.reduce((acc, item) => {
      const key = typeof groupBy === 'function' ? groupBy(item) : item[groupBy];
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(item);
      return acc;
    }, {} as Record<string, any[]>);

    return Object.entries(groups).map(([key, items]) => {
      const result: Record<string, any> = { [groupBy as string]: key };
      
      metrics.forEach(metric => {
        switch (metric.type) {
          case 'sum':
            result[metric.name] = items.reduce((sum, item) => sum + item[metric.field!], 0);
            break;
          case 'average':
            result[metric.name] = items.reduce((sum, item) => sum + item[metric.field!], 0) / items.length;
            break;
          case 'count':
            result[metric.name] = items.length;
            break;
          case 'custom':
            result[metric.name] = metric.fn!(items);
            break;
        }
      });

      return result;
    });
  }

  static transformData(data: any[], transformation: TransformationConfig): any[] {
    switch (transformation.type) {
      case 'normalize':
        return this.normalizeData(data, transformation.fields || []);
      case 'pivot':
        return this.pivotData(data, transformation.config);
      case 'timeseries':
        return this.timeseriesTransform(data, transformation.config);
      default:
        return data;
    }
  }

  private static normalizeData(data: any[], fields: string[]): any[] {
    const fieldStats = fields.reduce((acc, field) => {
      const values = data.map(item => item[field]);
      acc[field] = {
        min: Math.min(...values),
        max: Math.max(...values)
      };
      return acc;
    }, {} as Record<string, { min: number; max: number }>);

    return data.map(item => {
      const normalized = { ...item };
      fields.forEach(field => {
        const { min, max } = fieldStats[field];
        normalized[field] = (item[field] - min) / (max - min);
      });
      return normalized;
    });
  }

  private static pivotData(data: any[], config: any): any[] {
    // Implementation for pivot transformation
    return [];
  }

  private static timeseriesTransform(data: any[], config: any): any[] {
    // Implementation for timeseries transformation
    return [];
  }
} 