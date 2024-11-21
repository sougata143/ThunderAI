class DataAggregationService {
  constructor() {
    this.aggregators = new Map();
    this.windowSizes = new Map();
  }

  registerAggregator(type, aggregator, windowSize) {
    this.aggregators.set(type, aggregator);
    this.windowSizes.set(type, windowSize);
  }

  aggregate(type, data, options = {}) {
    const aggregator = this.aggregators.get(type);
    if (!aggregator) {
      throw new Error(`No aggregator registered for type: ${type}`);
    }

    const windowSize = options.windowSize || this.windowSizes.get(type);
    const now = Date.now();
    const windowStart = now - windowSize;

    // Filter data within time window
    const windowData = data.filter(point => point.timestamp >= windowStart);

    // Apply aggregation
    return aggregator(windowData, options);
  }

  // Built-in aggregators
  static sum(data, field) {
    return data.reduce((sum, point) => sum + point[field], 0);
  }

  static average(data, field) {
    if (data.length === 0) return 0;
    return DataAggregationService.sum(data, field) / data.length;
  }

  static count(data, predicate) {
    if (predicate) {
      return data.filter(predicate).length;
    }
    return data.length;
  }

  static groupBy(data, keyFn, valueFn) {
    return data.reduce((groups, point) => {
      const key = keyFn(point);
      if (!groups[key]) {
        groups[key] = [];
      }
      groups[key].push(valueFn ? valueFn(point) : point);
      return groups;
    }, {});
  }

  static movingAverage(data, field, window) {
    const result = [];
    for (let i = 0; i < data.length; i++) {
      const start = Math.max(0, i - window + 1);
      const windowData = data.slice(start, i + 1);
      result.push({
        ...data[i],
        [field]: DataAggregationService.average(windowData, field)
      });
    }
    return result;
  }
}

export default new DataAggregationService(); 