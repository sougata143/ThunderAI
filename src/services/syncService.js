import CollaborationService from './collaborationService';
import ConflictResolutionService from './conflictResolutionService';

class SyncService {
  constructor() {
    this.localChanges = new Map();
    this.syncInProgress = false;
    this.retryTimeout = 1000; // Start with 1 second retry
    this.maxRetryTimeout = 30000; // Max 30 seconds between retries
  }

  async syncChanges(documentId, changes) {
    try {
      // Store local changes
      if (!this.localChanges.has(documentId)) {
        this.localChanges.set(documentId, []);
      }
      this.localChanges.get(documentId).push(...changes);

      if (this.syncInProgress) return;
      this.syncInProgress = true;

      // Get all pending local changes
      const pendingChanges = this.localChanges.get(documentId) || [];
      
      // Send changes to server
      const response = await CollaborationService.syncChanges(documentId, pendingChanges);
      
      // Handle server response
      if (response.conflicts) {
        // Resolve conflicts using operational transform
        const resolvedOperations = ConflictResolutionService.resolveConflicts(
          documentId,
          [...pendingChanges, ...response.conflicts]
        );

        // Apply resolved operations locally
        this.applyResolvedOperations(documentId, resolvedOperations);
        
        // Sync resolved state with server
        await CollaborationService.updateDocument(documentId, {
          operations: resolvedOperations,
          version: response.serverVersion
        });
      }

      // Clear synced changes
      this.localChanges.set(documentId, []);
      this.retryTimeout = 1000; // Reset retry timeout on success
      this.syncInProgress = false;

    } catch (error) {
      console.error('Sync failed:', error);
      this.syncInProgress = false;
      
      // Exponential backoff for retries
      setTimeout(() => {
        this.syncChanges(documentId, []);
      }, this.retryTimeout);
      
      this.retryTimeout = Math.min(this.retryTimeout * 2, this.maxRetryTimeout);
    }
  }

  applyResolvedOperations(documentId, operations) {
    operations.forEach(operation => {
      switch (operation.type) {
        case 'update':
          this.applyUpdate(documentId, operation);
          break;
        case 'delete':
          this.applyDelete(documentId, operation);
          break;
        case 'insert':
          this.applyInsert(documentId, operation);
          break;
      }
    });
  }

  applyUpdate(documentId, operation) {
    // Apply update operation to local state
    const document = this.getDocument(documentId);
    if (document) {
      document.content = operation.content;
      document.version = operation.version;
      this.notifyListeners(documentId, 'update', document);
    }
  }

  applyDelete(documentId, operation) {
    // Apply delete operation to local state
    const document = this.getDocument(documentId);
    if (document) {
      document.deleted = true;
      document.version = operation.version;
      this.notifyListeners(documentId, 'delete', document);
    }
  }

  applyInsert(documentId, operation) {
    // Apply insert operation to local state
    const document = {
      id: documentId,
      content: operation.content,
      version: operation.version,
      metadata: operation.metadata
    };
    this.setDocument(documentId, document);
    this.notifyListeners(documentId, 'insert', document);
  }

  // Document storage and listener methods
  private documents = new Map();
  private listeners = new Map();

  getDocument(documentId) {
    return this.documents.get(documentId);
  }

  setDocument(documentId, document) {
    this.documents.set(documentId, document);
  }

  addListener(documentId, callback) {
    if (!this.listeners.has(documentId)) {
      this.listeners.set(documentId, new Set());
    }
    this.listeners.get(documentId).add(callback);
  }

  removeListener(documentId, callback) {
    if (this.listeners.has(documentId)) {
      this.listeners.get(documentId).delete(callback);
    }
  }

  notifyListeners(documentId, event, data) {
    if (this.listeners.has(documentId)) {
      this.listeners.get(documentId).forEach(callback => callback(event, data));
    }
  }
}

export default new SyncService(); 