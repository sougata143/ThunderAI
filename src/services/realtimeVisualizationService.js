import { io } from 'socket.io-client';

class RealtimeVisualizationService {
  constructor() {
    this.socket = null;
    this.subscriptions = new Map();
    this.dataBuffer = new Map();
    this.updateInterval = 1000; // Update interval in ms
    this.bufferSize = 100; // Maximum number of data points to keep
  }

  connect(visualizationId) {
    this.socket = io(`${process.env.REACT_APP_WEBSOCKET_URL}/visualizations`, {
      query: { visualizationId }
    });

    this.setupEventListeners();
  }

  setupEventListeners() {
    this.socket.on('data_update', ({ type, data }) => {
      // Buffer the data
      if (!this.dataBuffer.has(type)) {
        this.dataBuffer.set(type, []);
      }
      
      const buffer = this.dataBuffer.get(type);
      buffer.push(...data);
      
      // Trim buffer if it exceeds maximum size
      if (buffer.length > this.bufferSize) {
        buffer.splice(0, buffer.length - this.bufferSize);
      }

      // Notify subscribers
      if (this.subscriptions.has(type)) {
        this.subscriptions.get(type).forEach(callback => callback(buffer));
      }
    });

    this.socket.on('error', (error) => {
      console.error('Visualization socket error:', error);
    });
  }

  subscribe(type, callback) {
    if (!this.subscriptions.has(type)) {
      this.subscriptions.set(type, new Set());
      // Request initial data
      this.socket?.emit('subscribe', { type });
    }
    this.subscriptions.get(type).add(callback);

    // Return current buffer if available
    if (this.dataBuffer.has(type)) {
      callback(this.dataBuffer.get(type));
    }

    return () => this.unsubscribe(type, callback);
  }

  unsubscribe(type, callback) {
    if (this.subscriptions.has(type)) {
      this.subscriptions.get(type).delete(callback);
      if (this.subscriptions.get(type).size === 0) {
        this.subscriptions.delete(type);
        this.socket?.emit('unsubscribe', { type });
      }
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.subscriptions.clear();
    this.dataBuffer.clear();
  }
}

export default new RealtimeVisualizationService(); 