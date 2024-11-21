class DataService {
  static filterData(data, filters) {
    return data.filter(item => {
      return Object.entries(filters).every(([key, filter]) => {
        if (!filter.enabled) return true;
        
        const value = item[key];
        switch (filter.type) {
          case 'range':
            return value >= filter.min && value <= filter.max;
          case 'categorical':
            return filter.values.includes(value);
          case 'search':
            return value.toLowerCase().includes(filter.value.toLowerCase());
          default:
            return true;
        }
      });
    });
  }

  static aggregateData(data, aggregation) {
    const { groupBy, metrics } = aggregation;
    
    const groups = data.reduce((acc, item) => {
      const key = typeof groupBy === 'function' ? groupBy(item) : item[groupBy];
      if (!acc[key]) {
        acc[key] = [];
      }
      acc[key].push(item);
      return acc;
    }, {});

    return Object.entries(groups).map(([key, items]) => {
      const result = { [groupBy]: key };
      
      metrics.forEach(metric => {
        switch (metric.type) {
          case 'sum':
            result[metric.name] = items.reduce((sum, item) => sum + item[metric.field], 0);
            break;
          case 'average':
            result[metric.name] = items.reduce((sum, item) => sum + item[metric.field], 0) / items.length;
            break;
          case 'count':
            result[metric.name] = items.length;
            break;
          case 'custom':
            result[metric.name] = metric.fn(items);
            break;
        }
      });

      return result;
    });
  }

  static transformData(data, transformation) {
    switch (transformation.type) {
      case 'normalize':
        return this.normalizeData(data, transformation.fields);
      case 'pivot':
        return this.pivotData(data, transformation.config);
      case 'timeseries':
        return this.timeseriesTransform(data, transformation.config);
      default:
        return data;
    }
  }

  private static normalizeData(data, fields) {
    const fieldStats = fields.reduce((acc, field) => {
      const values = data.map(item => item[field]);
      acc[field] = {
        min: Math.min(...values),
        max: Math.max(...values)
      };
      return acc;
    }, {});

    return data.map(item => {
      const normalized = { ...item };
      fields.forEach(field => {
        const { min, max } = fieldStats[field];
        normalized[field] = (item[field] - min) / (max - min);
      });
      return normalized;
    });
  }

  private static pivotData(data, config) {
    // Implementation for pivot transformation
  }

  private static timeseriesTransform(data, config) {
    // Implementation for timeseries transformation
  }
}

export default DataService; 