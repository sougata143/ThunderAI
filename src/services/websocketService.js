class WebSocketService {
    constructor() {
        this.ws = null;
        this.subscribers = new Map();
    }

    connect(modelId) {
        const wsUrl = `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'}/training/${modelId}`;
        this.ws = new WebSocket(wsUrl);

        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            this.notifySubscribers(data);
        };

        this.ws.onerror = (error) => {
            console.error('WebSocket error:', error);
            this.reconnect(modelId);
        };

        this.ws.onclose = () => {
            console.log('WebSocket connection closed');
            this.reconnect(modelId);
        };
    }

    subscribe(callback) {
        const id = Math.random().toString(36).substr(2, 9);
        this.subscribers.set(id, callback);
        return id;
    }

    unsubscribe(id) {
        this.subscribers.delete(id);
    }

    notifySubscribers(data) {
        this.subscribers.forEach(callback => callback(data));
    }

    reconnect(modelId) {
        setTimeout(() => {
            console.log('Attempting to reconnect...');
            this.connect(modelId);
        }, 5000);
    }

    disconnect() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

export const wsService = new WebSocketService(); 