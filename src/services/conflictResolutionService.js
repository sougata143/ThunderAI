import { v4 as uuidv4 } from 'uuid';

class ConflictResolutionService {
  constructor() {
    this.operations = new Map();
    this.versions = new Map();
    this.pendingOperations = new Map();
  }

  // Operational Transform (OT) implementation
  transform(operation1, operation2) {
    // Transform two concurrent operations to preserve intention
    if (operation1.type === 'update' && operation2.type === 'update') {
      return this.transformUpdates(operation1, operation2);
    }
    // Add more transformation rules for different operation types
    return [operation1, operation2];
  }

  transformUpdates(op1, op2) {
    const timestamp1 = new Date(op1.timestamp).getTime();
    const timestamp2 = new Date(op2.timestamp).getTime();

    // Last-write-wins conflict resolution
    if (timestamp1 === timestamp2) {
      // If timestamps are equal, use client ID to break the tie
      return timestamp1 > timestamp2 ? [op1, null] : [null, op2];
    }

    return timestamp1 > timestamp2 ? [op1, null] : [null, op2];
  }

  applyOperation(documentId, operation) {
    const currentVersion = this.versions.get(documentId) || 0;
    operation.version = currentVersion + 1;
    operation.id = uuidv4();

    const pendingOps = this.pendingOperations.get(documentId) || [];
    
    // Transform against all pending operations
    let transformedOp = operation;
    for (const pendingOp of pendingOps) {
      const [newOp, _] = this.transform(transformedOp, pendingOp);
      if (newOp) transformedOp = newOp;
    }

    this.operations.set(operation.id, transformedOp);
    this.versions.set(documentId, operation.version);
    
    return transformedOp;
  }

  resolveConflicts(documentId, operations) {
    const sortedOps = operations.sort((a, b) => {
      const timeA = new Date(a.timestamp).getTime();
      const timeB = new Date(b.timestamp).getTime();
      return timeA - timeB;
    });

    const resolvedOps = [];
    for (const op of sortedOps) {
      const transformedOp = this.applyOperation(documentId, op);
      if (transformedOp) resolvedOps.push(transformedOp);
    }

    return resolvedOps;
  }

  getOperationHistory(documentId) {
    return Array.from(this.operations.values())
      .filter(op => op.documentId === documentId)
      .sort((a, b) => a.version - b.version);
  }
}

export default new ConflictResolutionService(); 