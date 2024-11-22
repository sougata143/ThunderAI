class WebSocketManager {
  constructor(url, options = {}) {
    this.url = url;
    this.options = {
      reconnectAttempts: 5,
      reconnectInterval: 3000,
      onMessage: () => {},
      onError: () => {},
      onClose: () => {},
      onOpen: () => {},
      ...options
    };
    this.reconnectCount = 0;
    this.isIntentionalClose = false;
    this.connect();
  }

  connect() {
    try {
      console.log(`Attempting to connect to WebSocket at: ${this.url}`);
      
      // Create WebSocket connection
      this.ws = new WebSocket(this.url);

      // Set connection timeout
      this.connectionTimeout = setTimeout(() => {
        if (this.ws.readyState !== WebSocket.OPEN) {
          console.error('WebSocket connection timeout');
          this.ws.close();
          this.handleError(new Error('Connection timeout'));
        }
      }, 10000);

      this.ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          console.log('WebSocket message received:', {
            type: typeof data,
            preview: JSON.stringify(data).slice(0, 100),
            timestamp: new Date().toISOString()
          });
          this.options.onMessage(event);
        } catch (error) {
          console.error('Error processing message:', error);
        }
      };

      this.ws.onerror = (error) => {
        console.error('WebSocket error details:', {
          error,
          readyState: this.ws.readyState,
          url: this.url,
          timestamp: new Date().toISOString(),
          reconnectCount: this.reconnectCount
        });
        this.handleError(error);
      };

      this.ws.onclose = (event) => {
        clearTimeout(this.connectionTimeout);
        console.log('WebSocket close event:', {
          code: event.code,
          reason: event.reason || 'No reason provided',
          wasClean: event.wasClean,
          isIntentionalClose: this.isIntentionalClose,
          timestamp: new Date().toISOString(),
          readyState: this.ws.readyState
        });

        // Check if backend is available
        this.checkBackendAvailability()
          .then(isAvailable => {
            if (!isAvailable) {
              this.options.onError(new Error('Backend service is not available'));
              return;
            }
            this.handleClose(event);
          });
      };

      this.ws.onopen = () => {
        clearTimeout(this.connectionTimeout);
        console.log('WebSocket connection established successfully');
        this.reconnectCount = 0;
        this.isIntentionalClose = false;
        this.options.onOpen();
      };
    } catch (error) {
      clearTimeout(this.connectionTimeout);
      console.error('Error during WebSocket connection setup:', error);
      this.handleError(error);
    }
  }

  async checkBackendAvailability() {
    try {
      // Try to fetch a health check endpoint
      const response = await fetch('/api/health');
      return response.ok;
    } catch (error) {
      console.error('Backend health check failed:', error);
      return false;
    }
  }

  handleError(error) {
    console.error('WebSocket error:', {
      error,
      timestamp: new Date().toISOString(),
      connectionState: this.ws?.readyState
    });
    this.options.onError(error);
  }

  handleClose(event) {
    this.options.onClose(event);
    
    if (!this.isIntentionalClose && event.code !== 1000) {
      console.log('Abnormal closure detected, checking connection status...');
      
      if (navigator.onLine) {
        console.log('Network is available, attempting to reconnect...');
        this.attemptReconnect();
      } else {
        console.log('Network is offline, waiting for online event...');
        window.addEventListener('online', () => {
          console.log('Network is back online, attempting to reconnect...');
          this.attemptReconnect();
        }, { once: true });
      }
    }
  }

  attemptReconnect() {
    if (this.reconnectCount < this.options.reconnectAttempts) {
      this.reconnectCount++;
      const backoffTime = Math.min(
        this.options.reconnectInterval * Math.pow(1.5, this.reconnectCount - 1),
        30000
      );
      
      console.log(`Scheduling reconnection attempt ${this.reconnectCount}/${this.options.reconnectAttempts} in ${backoffTime}ms`);
      
      setTimeout(() => {
        if (navigator.onLine) {
          console.log(`Executing reconnection attempt ${this.reconnectCount}`);
          this.connect();
        } else {
          console.log('Network still offline, skipping reconnection attempt');
        }
      }, backoffTime);
    } else {
      console.error('Max reconnection attempts reached');
      this.options.onError(new Error('Max reconnection attempts reached'));
    }
  }

  send(data) {
    if (this.ws?.readyState === WebSocket.OPEN) {
      try {
        const message = typeof data === 'string' ? data : JSON.stringify(data);
        this.ws.send(message);
      } catch (error) {
        console.error('Error sending WebSocket message:', error);
        this.handleError(error);
      }
    } else {
      console.warn('WebSocket is not open. Message not sent. ReadyState:', this.ws?.readyState);
    }
  }

  close() {
    clearTimeout(this.connectionTimeout);
    
    if (this.ws) {
      this.isIntentionalClose = true;
      console.log('Intentionally closing WebSocket connection');
      this.ws.close(1000, 'Intentional close');
    }
  }
}

export default WebSocketManager; 