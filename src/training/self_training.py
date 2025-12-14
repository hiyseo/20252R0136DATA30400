"""
Self-training for semi-supervised learning.
Iterative pseudo-labeling and retraining.
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List, Dict, Optional
from tqdm import tqdm
import copy


class SelfTrainer:
    """Self-training with pseudo-labeling."""
    
    def __init__(self,
                 model: nn.Module,
                 device: torch.device,
                 confidence_threshold: float = 0.7,
                 max_iterations: int = 3,
                 min_improvement: float = 0.001):
        """
        Initialize self-trainer.
        
        Args:
            model: Base model to train
            device: Device to use
            confidence_threshold: Minimum confidence for pseudo-labels
            max_iterations: Maximum self-training iterations
            min_improvement: Minimum improvement to continue
        """
        self.model = model
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.max_iterations = max_iterations
        self.min_improvement = min_improvement
    
    @torch.no_grad()
    def generate_pseudo_labels(self,
                              unlabeled_loader: DataLoader,
                              use_amp: bool = False,
                              use_soft_labels: bool = True,
                              blend_with_silver: bool = True,
                              silver_weight: float = 0.5) -> Tuple[List, List, List]:
        """
        Generate pseudo-labels for unlabeled data.
        
        Args:
            unlabeled_loader: DataLoader for unlabeled data
            use_amp: Use automatic mixed precision
            use_soft_labels: Use soft probability labels instead of hard binary
            blend_with_silver: Blend model predictions with silver labels if available
            silver_weight: Weight for silver labels when blending (0-1)
            
        Returns:
            Tuple of (input_ids, attention_masks, pseudo_labels)
        """
        self.model.eval()
        
        all_input_ids = []
        all_attention_masks = []
        all_pseudo_labels = []
        
        for batch in tqdm(unlabeled_loader, desc="Generating pseudo-labels"):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Check if silver labels are available
            has_silver_labels = 'labels' in batch and 'confidences' in batch
            
            # Predict
            if use_amp:
                from torch.cuda.amp import autocast
                with autocast():
                    logits = self.model(input_ids, attention_mask)
            else:
                logits = self.model(input_ids, attention_mask)
            
            probs = torch.sigmoid(logits)
            
            # Blend with silver labels if available
            if has_silver_labels and blend_with_silver:
                silver_labels = batch['labels'].to(self.device)
                silver_confidences = batch['confidences'].to(self.device)
                
                # Weighted blend: silver labels + model predictions
                blended_probs = silver_weight * silver_labels + (1 - silver_weight) * probs
                probs = blended_probs
            
            # Filter by confidence: at least one prediction above threshold
            max_probs = probs.max(dim=1)[0]
            has_confident = max_probs >= self.confidence_threshold
            
            if has_confident.any():
                all_input_ids.append(input_ids[has_confident].cpu())
                all_attention_masks.append(attention_mask[has_confident].cpu())
                
                # Use soft labels (probabilities) or hard labels (binary)
                if use_soft_labels:
                    # Keep probability distribution as soft targets
                    pseudo_labels = probs[has_confident].cpu()
                else:
                    # Hard binary labels
                    pseudo_labels = (probs[has_confident] >= 0.5).float().cpu()
                
                all_pseudo_labels.append(pseudo_labels)
        
        if len(all_input_ids) == 0:
            return [], [], []
        
        # Concatenate with safety checks
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        all_pseudo_labels = torch.cat(all_pseudo_labels, dim=0)
        
        # Ensure proper dtype and shapes to avoid out of range errors
        all_input_ids = all_input_ids.long()  # Ensure int64
        all_attention_masks = all_attention_masks.long()  # Ensure int64
        all_pseudo_labels = all_pseudo_labels.float()  # Ensure float32
        
        # Clamp input_ids to valid range to prevent out of range errors
        vocab_size = 30522  # BERT vocab size
        all_input_ids = torch.clamp(all_input_ids, 0, vocab_size - 1)
        
        return all_input_ids, all_attention_masks, all_pseudo_labels
    
    def create_combined_loader(self,
                              labeled_loader: DataLoader,
                              pseudo_input_ids: torch.Tensor,
                              pseudo_attention_masks: torch.Tensor,
                              pseudo_labels: torch.Tensor,
                              batch_size: int = 16,
                              sample_weight: float = 0.5) -> DataLoader:
        """
        Combine labeled and pseudo-labeled data.
        
        Args:
            labeled_loader: Original labeled data loader
            pseudo_input_ids: Pseudo-labeled input IDs
            pseudo_attention_masks: Pseudo-labeled attention masks
            pseudo_labels: Pseudo labels
            batch_size: Batch size
            
        Returns:
            Combined DataLoader
        """
        # Extract labeled data
        labeled_data = []
        for batch in labeled_loader:
            labeled_data.append({
                'input_ids': batch['input_ids'],
                'attention_mask': batch['attention_mask'],
                'labels': batch['labels']
            })
        
        # Combine
        all_input_ids = [batch['input_ids'] for batch in labeled_data]
        all_attention_masks = [batch['attention_mask'] for batch in labeled_data]
        all_labels = [batch['labels'] for batch in labeled_data]
        
        if len(pseudo_input_ids) > 0:
            all_input_ids.append(pseudo_input_ids)
            all_attention_masks.append(pseudo_attention_masks)
            all_labels.append(pseudo_labels)
        
        all_input_ids = torch.cat(all_input_ids, dim=0)
        all_attention_masks = torch.cat(all_attention_masks, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        # Ensure proper dtype for TensorDataset
        all_input_ids = all_input_ids.long()
        all_attention_masks = all_attention_masks.long()
        all_labels = all_labels.float()
        
        # Create new dataset
        dataset = TensorDataset(all_input_ids, all_attention_masks, all_labels)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        return loader
    
    def train_iteration(self,
                       train_loader: DataLoader,
                       criterion: nn.Module,
                       optimizer: torch.optim.Optimizer,
                       scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                       num_epochs: int = 1,
                       use_amp: bool = False,
                       gradient_clip_norm: float = 1.0) -> float:
        """
        Train for one self-training iteration.
        
        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs: Number of epochs
            use_amp: Use automatic mixed precision
            gradient_clip_norm: Gradient clipping norm
            
        Returns:
            Average training loss
        """
        self.model.train()
        
        scaler = None
        if use_amp:
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
        
        total_loss = 0
        num_batches = 0
        
        for epoch in range(num_epochs):
            pbar = tqdm(train_loader, desc=f"Training epoch {epoch+1}/{num_epochs}")
            
            for batch in pbar:
                if len(batch) == 3:  # TensorDataset format
                    input_ids, attention_mask, labels = batch
                    input_ids = input_ids.to(self.device)
                    attention_mask = attention_mask.to(self.device)
                    labels = labels.to(self.device)
                else:  # Dict format
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                if use_amp and scaler is not None:
                    from torch.cuda.amp import autocast
                    with autocast():
                        logits = self.model(input_ids, attention_mask)
                        loss = criterion(logits, labels)
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logits = self.model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), gradient_clip_norm)
                    optimizer.step()
                
                if scheduler is not None:
                    scheduler.step()
                
                total_loss += loss.item()
                num_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def self_train(self,
                   labeled_loader: DataLoader,
                   unlabeled_loader: DataLoader,
                   criterion: nn.Module,
                   optimizer: torch.optim.Optimizer,
                   scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
                   num_epochs_per_iteration: int = 1,
                   use_amp: bool = False,
                   logger = None) -> Dict:
        """
        Perform self-training.
        
        Args:
            labeled_loader: Labeled data loader
            unlabeled_loader: Unlabeled data loader
            criterion: Loss function
            optimizer: Optimizer
            scheduler: Learning rate scheduler
            num_epochs_per_iteration: Epochs per iteration
            use_amp: Use automatic mixed precision
            logger: Logger object
            
        Returns:
            Training statistics
        """
        stats = {
            'iterations': [],
            'losses': [],
            'pseudo_label_counts': []
        }
        
        prev_loss = float('inf')
        
        for iteration in range(self.max_iterations):
            if logger:
                logger.info(f"\n{'='*60}")
                logger.info(f"Self-training Iteration {iteration + 1}/{self.max_iterations}")
                logger.info(f"{'='*60}")
            else:
                print(f"\n{'='*60}")
                print(f"Self-training Iteration {iteration + 1}/{self.max_iterations}")
                print(f"{'='*60}")
            
            # Generate pseudo-labels (soft labels)
            pseudo_ids, pseudo_masks, pseudo_labels = self.generate_pseudo_labels(
                unlabeled_loader, use_amp, use_soft_labels=True
            )
            
            num_pseudo = len(pseudo_ids) if len(pseudo_ids) > 0 else 0
            
            if logger:
                logger.info(f"Generated {num_pseudo} pseudo-labeled samples")
            else:
                print(f"Generated {num_pseudo} pseudo-labeled samples")
            
            if num_pseudo == 0:
                if logger:
                    logger.info("No confident pseudo-labels generated. Stopping.")
                else:
                    print("No confident pseudo-labels generated. Stopping.")
                break
            
            # Combine data
            combined_loader = self.create_combined_loader(
                labeled_loader,
                pseudo_ids,
                pseudo_masks,
                pseudo_labels,
                batch_size=labeled_loader.batch_size
            )
            
            # Train
            avg_loss = self.train_iteration(
                combined_loader,
                criterion,
                optimizer,
                scheduler,
                num_epochs_per_iteration,
                use_amp
            )
            
            if logger:
                logger.info(f"Iteration {iteration + 1} loss: {avg_loss:.4f}")
            else:
                print(f"Iteration {iteration + 1} loss: {avg_loss:.4f}")
            
            # Record stats
            stats['iterations'].append(iteration + 1)
            stats['losses'].append(avg_loss)
            stats['pseudo_label_counts'].append(num_pseudo)
            
            # Check improvement
            improvement = prev_loss - avg_loss
            if improvement < self.min_improvement:
                if logger:
                    logger.info(f"Improvement {improvement:.4f} < {self.min_improvement}. Stopping.")
                else:
                    print(f"Improvement {improvement:.4f} < {self.min_improvement}. Stopping.")
                break
            
            prev_loss = avg_loss
        
        return stats
