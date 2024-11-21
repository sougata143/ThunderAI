import { io } from 'socket.io-client';
import SyncService from './syncService';
import ConflictResolutionService from './conflictResolutionService';

class CollaborationService {
  constructor() {
    this.socket = null;
    this.collaborators = new Map();
    this.cursorPositions = new Map();
    this.listeners = new Map();
  }

  connect(dashboardId, userId, userName) {
    this.socket = io(process.env.REACT_APP_WEBSOCKET_URL, {
      query: { dashboardId, userId, userName }
    });

    this.setupEventListeners();
  }

  setupEventListeners() {
    this.socket.on('user_joined', (user) => {
      this.collaborators.set(user.id, user);
      this.notifyListeners('collaborators_changed', Array.from(this.collaborators.values()));
    });

    this.socket.on('user_left', (userId) => {
      this.collaborators.delete(userId);
      this.cursorPositions.delete(userId);
      this.notifyListeners('collaborators_changed', Array.from(this.collaborators.values()));
    });

    this.socket.on('cursor_moved', ({ userId, position }) => {
      this.cursorPositions.set(userId, position);
      this.notifyListeners('cursor_moved', { userId, position });
    });

    this.socket.on('widget_updated', (update) => {
      this.notifyListeners('widget_updated', update);
    });

    this.socket.on('layout_changed', (layout) => {
      this.notifyListeners('layout_changed', layout);
    });

    this.socket.on('document_changed', ({ documentId, changes }) => {
      // Handle remote changes
      const resolvedChanges = ConflictResolutionService.resolveConflicts(
        documentId,
        changes
      );
      SyncService.applyResolvedOperations(documentId, resolvedChanges);
    });

    this.socket.on('sync_required', ({ documentId }) => {
      // Handle sync request from server
      this.syncDocument(documentId);
    });
  }

  updateCursorPosition(position) {
    if (this.socket) {
      this.socket.emit('cursor_moved', position);
    }
  }

  updateWidget(widgetId, update) {
    if (this.socket) {
      this.socket.emit('widget_updated', { widgetId, update });
    }
  }

  updateLayout(layout) {
    if (this.socket) {
      this.socket.emit('layout_changed', layout);
    }
  }

  subscribe(event, callback) {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event).add(callback);
  }

  unsubscribe(event, callback) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).delete(callback);
    }
  }

  notifyListeners(event, data) {
    if (this.listeners.has(event)) {
      this.listeners.get(event).forEach(callback => callback(data));
    }
  }

  disconnect() {
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
    this.collaborators.clear();
    this.cursorPositions.clear();
    this.listeners.clear();
  }

  async syncDocument(documentId) {
    try {
      const localChanges = await SyncService.getLocalChanges(documentId);
      await SyncService.syncChanges(documentId, localChanges);
    } catch (error) {
      console.error('Error syncing document:', error);
    }
  }
}

export default new CollaborationService(); 