export interface ModelConfig {
  modelType: string;
  numLabels: number;
  maxLength: number;
  batchSize: number;
  learningRate: number;
  epochs: number;
  deviceType: 'cpu' | 'cuda' | 'mps';
  [key: string]: any;
}

export interface ModelMetadata {
  name: string;
  version: string;
  createdAt: Date;
  updatedAt: Date;
  metrics: {
    accuracy?: number;
    loss?: number;
    [key: string]: any;
  };
}

export interface PredictionResult {
  label: number | string;
  confidence: number;
  probabilities: number[];
  latency: number;
}

export interface BaseModelInterface {
  initialize(): Promise<void>;
  train(data: any): Promise<ModelMetadata>;
  predict(input: any): Promise<PredictionResult>;
  save(path: string): Promise<void>;
  load(path: string): Promise<void>;
  getMetadata(): ModelMetadata;
} 