"""
Baseline model training script.
"""

import os
import sys
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path

from src.data_preprocessing import DataLoader as DataPreprocessor
from src.dataset import create_dataloaders
from src.models.encoder import TextEncoder, get_device
from src.training.loss_functions import get_loss_function
from src.training.self_training import SelfTrainer
from src.utils.metrics import compute_multilabel_metrics, find_optimal_threshold
from src.utils.seed import set_seed
from src.silver_labeling.generate_silver_labels import SilverLabelGenerator


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / num_batches


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    """Evaluate model."""
    model.eval()
    total_loss = 0
    all_logits = []
    all_labels = []
    
    for batch in tqdm(dataloader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        total_loss += loss.item()
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to numpy
    scores = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()
    
    # Find optimal threshold
    threshold, best_f1 = find_optimal_threshold(labels, scores, metric='micro_f1')
    
    # Compute metrics with optimal threshold
    predictions = (scores >= threshold).astype(int)
    metrics = compute_multilabel_metrics(labels, predictions)
    metrics['threshold'] = threshold
    metrics['loss'] = total_loss / len(dataloader)
    
    return metrics


def train_baseline_model(args):
    """Main training function with optional self-training."""
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("\n=== Loading Data ===")
    data_loader = DataPreprocessor()
    data_loader.load_all()
    
    # Load silver labels
    print("Loading silver labels...")
    train_labels, train_confidences = SilverLabelGenerator.load_labels(
        args.train_labels_path
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        data_loader.train_corpus,
        data_loader.test_corpus,
        train_labels,
        train_confidences,
        tokenizer_name=args.model_name,
        batch_size=args.batch_size,
        max_length=args.max_length,
        num_workers=args.num_workers
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print(f"\n=== Creating Model: {args.model_name} ===")
    model = TextEncoder(
        model_name=args.model_name,
        num_classes=len(data_loader.classes),
        dropout=args.dropout,
        freeze_encoder=args.freeze_encoder
    ).to(device)
    
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Loss function
    criterion = get_loss_function(args.loss_type)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Scheduler
    total_steps = len(train_loader) * args.num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps
    )
    
    # Self-training setup (2-stage training)
    if args.use_self_training:
        print(f"\n=== Self-Training Enabled (2-Stage) ===")
        print(f"Confidence threshold: {args.self_training_confidence}")
        print(f"Max iterations: {args.self_training_iterations}")
        
        # ========================================
        # STAGE 1: Initial training with BCE loss
        # ========================================
        print(f"\n=== Stage 1: Initial Training with BCE Loss ===")
        print(f"Training for {args.num_epochs} epochs to initialize model...")
        
        training_history = {'train_loss': [], 'epochs': []}
        
        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            
            # Train with BCE loss
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            print(f"Train loss (BCE): {train_loss:.4f}")
            
            training_history['train_loss'].append(train_loss)
            training_history['epochs'].append(epoch + 1)
        
        print(f"\n✓ Stage 1 completed - Model initialized with BCE loss")
        
        # ========================================
        # STAGE 2: Self-training with KLD loss
        # ========================================
        print(f"\n=== Stage 2: Self-Training with KLD Loss ===")
        
        self_trainer = SelfTrainer(
            model=model,
            device=device,
            confidence_threshold=args.self_training_confidence,
            max_iterations=args.self_training_iterations,
            min_improvement=0.001
        )
        
        # Use test_loader as unlabeled data for pseudo-labeling
        unlabeled_loader = test_loader
        
        # Self-training with KLD loss for soft pseudo-labels
        kld_criterion = get_loss_function('kld')
        
        stats = self_trainer.self_train(
            labeled_loader=train_loader,
            unlabeled_loader=unlabeled_loader,
            criterion=kld_criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            num_epochs_per_iteration=args.num_epochs,
            use_amp=False
        )
        
        print(f"\n✓ Stage 2 completed - Self-training finished")
        print(f"Self-training iterations: {len(stats['iterations'])}")
        print(f"Final loss: {stats['losses'][-1]:.4f}")
        
        # Save combined training history
        import json
        training_history['self_training_stats'] = stats
        history_path = Path(args.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
    else:
        # Standard training loop (BCE only)
        print(f"\n=== Standard Training (BCE Loss) ===")
        print(f"Training for {args.num_epochs} epochs...")
        
        training_history = {'train_loss': [], 'epochs': []}
        
        for epoch in range(args.num_epochs):
            print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
            
            # Train with BCE loss
            train_loss = train_epoch(model, train_loader, criterion, optimizer, scheduler, device)
            print(f"Train loss: {train_loss:.4f}")
            
            training_history['train_loss'].append(train_loss)
            training_history['epochs'].append(epoch + 1)
            
            # Save checkpoint
            if (epoch + 1) % args.save_every == 0:
                checkpoint_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_loss': train_loss,
                }, checkpoint_path)
                
                print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Save training history
        import json
        history_path = Path(args.output_dir) / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
    
    # Save final model
    final_model_path = Path(args.output_dir) / "best_model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved: {final_model_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_labels_path', type=str, 
                       default='data/intermediate/train_silver_labels.pkl')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model
    parser.add_argument('--model_name', type=str, default='bert-base-uncased')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--freeze_encoder', action='store_true')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=3)
    parser.add_argument('--learning_rate', type=float, default=2e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--loss_type', type=str, default='bce', 
                       choices=['bce', 'focal', 'asymmetric', 'kld'])
    parser.add_argument('--model_type', type=str, default='baseline',
                       help='Model type for directory naming (baseline, gcn, gat, etc.)')
    
    # Self-training
    parser.add_argument('--use_self_training', action='store_true',
                       help='Enable self-training with pseudo-labels')
    parser.add_argument('--self_training_confidence', type=float, default=0.7,
                       help='Confidence threshold for pseudo-labels')
    parser.add_argument('--self_training_iterations', type=int, default=3,
                       help='Maximum self-training iterations')
    
    # Misc
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory. If not specified, defaults to models/{model_type}')
    parser.add_argument('--save_every', type=int, default=1)
    
    args = parser.parse_args()
    
    # Set output_dir based on model_type if not specified
    if args.output_dir is None:
        args.output_dir = f'models/{args.model_type}'
    
    # Train
    train_baseline_model(args)


if __name__ == "__main__":
    main()
