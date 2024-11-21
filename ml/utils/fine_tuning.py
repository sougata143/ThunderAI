from typing import Dict, Any, List, Optional, Callable
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import PreTrainedModel, PreTrainedTokenizer, get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import logging

class FineTuningConfig:
    def __init__(
        self,
        num_epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        evaluation_strategy: str = "epoch",
        save_strategy: str = "epoch",
        metric_for_best_model: str = "loss",
        greater_is_better: bool = False
    ):
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.warmup_steps = warmup_steps
        self.weight_decay = weight_decay
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.evaluation_strategy = evaluation_strategy
        self.save_strategy = save_strategy
        self.metric_for_best_model = metric_for_best_model
        self.greater_is_better = greater_is_better

class FineTuner:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        config: FineTuningConfig,
        train_dataset: Dataset,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics: Optional[Callable] = None
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.compute_metrics = compute_metrics
        
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True
        )
        
        if eval_dataset:
            self.eval_dataloader = DataLoader(
                eval_dataset,
                batch_size=config.batch_size
            )
        
        # Set up optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Initialize tracking variables
        self.best_metric = float('inf')
        if config.greater_is_better:
            self.best_metric = float('-inf')
        
        self.global_step = 0
        
    def _create_optimizer(self):
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                'weight_decay': self.config.weight_decay
            },
            {
                'params': [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]
        return torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate
        )
    
    def _create_scheduler(self):
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=len(self.train_dataloader) * self.config.num_epochs
        )
    
    def train(self) -> Dict[str, List[float]]:
        """Fine-tune the model"""
        training_history = {
            'train_loss': [],
            'eval_loss': [],
            'metrics': []
        }
        
        for epoch in range(self.config.num_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for step, batch in enumerate(tqdm(self.train_dataloader)):
                outputs = self.model(**batch)
                loss = outputs.loss
                
                # Gradient accumulation
                loss = loss / self.config.gradient_accumulation_steps
                loss.backward()
                
                if (step + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm
                    )
                    
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    self.global_step += 1
                
                train_loss += loss.item()
            
            avg_train_loss = train_loss / len(self.train_dataloader)
            training_history['train_loss'].append(avg_train_loss)
            
            # Evaluation
            if self.eval_dataset and self.config.evaluation_strategy == "epoch":
                eval_results = self.evaluate()
                training_history['eval_loss'].append(eval_results['eval_loss'])
                training_history['metrics'].append(eval_results['metrics'])
                
                # Save best model
                current_metric = eval_results['metrics'][self.config.metric_for_best_model]
                if self._is_better_metric(current_metric):
                    self.best_metric = current_metric
                    self.save_model('best_model')
            
            # Save checkpoint
            if self.config.save_strategy == "epoch":
                self.save_model(f'checkpoint-{epoch}')
        
        return training_history
    
    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the model"""
        self.model.eval()
        eval_loss = 0
        eval_predictions = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                outputs = self.model(**batch)
                loss = outputs.loss
                eval_loss += loss.item()
                
                predictions = outputs.logits.argmax(dim=-1)
                eval_predictions.extend(predictions.tolist())
                eval_labels.extend(batch['labels'].tolist())
        
        avg_eval_loss = eval_loss / len(self.eval_dataloader)
        
        results = {
            'eval_loss': avg_eval_loss,
            'metrics': {}
        }
        
        if self.compute_metrics:
            metrics = self.compute_metrics(eval_predictions, eval_labels)
            results['metrics'] = metrics
        
        return results
    
    def _is_better_metric(self, current_metric: float) -> bool:
        if self.config.greater_is_better:
            return current_metric > self.best_metric
        return current_metric < self.best_metric
    
    def save_model(self, output_dir: str):
        """Save model checkpoint"""
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir) 