export interface ConfusionMatrix {
  truePositives: number;
  trueNegatives: number;
  falsePositives: number;
  falseNegatives: number;
}

export interface ROCPoint {
  falsePositiveRate: number;
  truePositiveRate: number;
  threshold: number;
}

export interface PRPoint {
  precision: number;
  recall: number;
  threshold: number;
}

export interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  confusionMatrix: ConfusionMatrix;
  rocCurve: ROCPoint[];
  prCurve: PRPoint[];
}

export interface EvaluationResult {
  modelId: string;
  timestamp: string;
  metrics: ModelMetrics;
  metadata?: Record<string, any>;
}

export interface ShapValue {
  feature: string;
  value: number;
  impact: number;
}

export interface LimeExplanation {
  feature: string;
  weight: number;
  confidence: number;
}

export interface PredictionExplanation {
  shapValues: ShapValue[];
  limeExplanation: LimeExplanation[];
} 