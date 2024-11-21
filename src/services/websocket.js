class WebSocketService {
  constructor(url) {
    this.url = url;
    this.ws = null;
    this.listeners = new Map();
  }

  connect() {
    this.ws = new WebSocket(this.url);

    this.ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      if (this.listeners.has(data.type)) {
        this.listeners.get(data.type).forEach(callback => callback(data.payload));
      }
    };

    this.ws.onclose = () => {
      setTimeout(() => {
        this.connect();
      }, 1000);
    };
  }

  subscribe(type, callback) {
    if (!this.listeners.has(type)) {
      this.listeners.set(type, new Set());
    }
    this.listeners.get(type).add(callback);
  }

  unsubscribe(type, callback) {
    if (this.listeners.has(type)) {
      this.listeners.get(type).delete(callback);
    }
  }
}

export default new WebSocketService('ws://localhost:8000/ws'); 