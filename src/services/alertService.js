import { Subject } from 'rxjs';

class AlertService {
  constructor() {
    this.alerts = new Subject();
    this.rules = new Map();
    this.activeAlerts = new Map();
    this.alertHistory = [];
  }

  addRule(ruleId, config) {
    const rule = {
      id: ruleId,
      condition: config.condition,
      threshold: config.threshold,
      severity: config.severity || 'info',
      message: config.message,
      cooldown: config.cooldown || 300000, // 5 minutes default
      lastTriggered: 0
    };
    this.rules.set(ruleId, rule);
  }

  removeRule(ruleId) {
    this.rules.delete(ruleId);
    this.activeAlerts.delete(ruleId);
  }

  checkMetric(metricName, value, metadata = {}) {
    this.rules.forEach((rule, ruleId) => {
      const now = Date.now();
      if (now - rule.lastTriggered < rule.cooldown) return;

      if (rule.condition(value, rule.threshold)) {
        const alert = {
          id: `${ruleId}-${now}`,
          ruleId,
          timestamp: now,
          value,
          metadata,
          severity: rule.severity,
          message: typeof rule.message === 'function' 
            ? rule.message(value, metadata)
            : rule.message
        };

        this.triggerAlert(alert);
        rule.lastTriggered = now;
      } else if (this.activeAlerts.has(ruleId)) {
        this.resolveAlert(ruleId);
      }
    });
  }

  triggerAlert(alert) {
    this.activeAlerts.set(alert.ruleId, alert);
    this.alertHistory.push({ ...alert, status: 'triggered' });
    this.alerts.next({
      type: 'triggered',
      alert
    });
  }

  resolveAlert(ruleId) {
    const alert = this.activeAlerts.get(ruleId);
    if (alert) {
      this.activeAlerts.delete(ruleId);
      this.alertHistory.push({ 
        ...alert, 
        status: 'resolved',
        resolvedAt: Date.now()
      });
      this.alerts.next({
        type: 'resolved',
        alert
      });
    }
  }

  subscribe(callback) {
    return this.alerts.subscribe(callback);
  }

  getActiveAlerts() {
    return Array.from(this.activeAlerts.values());
  }

  getAlertHistory(options = {}) {
    let history = [...this.alertHistory];
    
    if (options.severity) {
      history = history.filter(alert => alert.severity === options.severity);
    }
    
    if (options.timeRange) {
      const cutoff = Date.now() - options.timeRange;
      history = history.filter(alert => alert.timestamp >= cutoff);
    }
    
    return history;
  }
}

export default new AlertService(); 