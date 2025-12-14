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
import json
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns

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


def plot_training_curves(training_history, training_dir, timestamp=None):
    """Plot and save training curves."""
    # Create training results directory
    training_dir = Path(training_dir)
    training_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate timestamp prefix if not provided
    if timestamp is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('husl')
    
    # Plot training loss
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if 'train_loss' in training_history and training_history['train_loss']:
        epochs = training_history.get('epochs', list(range(1, len(training_history['train_loss']) + 1)))
        ax.plot(epochs, training_history['train_loss'], 'o-', linewidth=2, markersize=6, label='Training Loss')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss Curve', fontsize=14, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        loss_curve_path = training_dir / f'training_{timestamp}_loss_curve.png'
        plt.savefig(loss_curve_path, dpi=300, bbox_inches='tight')
        print(f"✓ Training loss curve saved: {loss_curve_path}")
        plt.close()
    
    # Plot self-training statistics if available
    if 'self_training_stats' in training_history:
        stats = training_history['self_training_stats']
        
        if 'iterations' in stats and 'losses' in stats and len(stats['losses']) > 0:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Self-training loss curve
            ax1.plot(stats['iterations'], stats['losses'], 'o-', linewidth=2, markersize=6, color='#e74c3c')
            ax1.set_xlabel('Iteration', fontsize=12)
            ax1.set_ylabel('Loss', fontsize=12)
            ax1.set_title('Self-Training Loss (KLD)', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Pseudo-label statistics
            if 'pseudo_label_counts' in stats and len(stats['pseudo_label_counts']) > 0:
                ax2.plot(stats['iterations'], stats['pseudo_label_counts'], 's-', linewidth=2, markersize=6, color='#3498db')
                ax2.set_xlabel('Iteration', fontsize=12)
                ax2.set_ylabel('Number of Pseudo-labels', fontsize=12)
                ax2.set_title('Pseudo-label Generation', fontsize=14, fontweight='bold')
                ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            self_training_path = training_dir / f'training_{timestamp}_self_training_curves.png'
            plt.savefig(self_training_path, dpi=300, bbox_inches='tight')
            print(f"✓ Self-training curves saved: {self_training_path}")
            plt.close()
        else:
            print(f"⚠ Self-training enabled but no iterations completed (no visualization generated)")
    
    # Combined plot for 2-stage training
    if 'train_loss' in training_history and 'self_training_stats' in training_history:
        stats = training_history['self_training_stats']
        
        # Only create combined plot if self-training actually ran
        if 'iterations' in stats and 'losses' in stats and len(stats['losses']) > 0:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Stage 1: BCE
            bce_epochs = training_history.get('epochs', [])
            bce_losses = training_history.get('train_loss', [])
            
            if bce_epochs and bce_losses:
                ax.plot(bce_epochs, bce_losses, 'o-', linewidth=2.5, markersize=7, 
                       color='#2ecc71', label='Stage 1: BCE Loss', alpha=0.8)
            
            # Stage 2: Self-training
            # Offset iterations to continue from BCE epochs
            offset = max(bce_epochs) if bce_epochs else 0
            st_iterations = [offset + i for i in stats['iterations']]
            ax.plot(st_iterations, stats['losses'], 's-', linewidth=2.5, markersize=7,
                   color='#e74c3c', label='Stage 2: Self-Training (KLD)', alpha=0.8)
            
            ax.set_xlabel('Training Step', fontsize=12)
            ax.set_ylabel('Loss', fontsize=12)
            ax.set_title('2-Stage Training: BCE → Self-Training', fontsize=14, fontweight='bold')
            ax.legend(fontsize=11, loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add vertical line to separate stages
            if bce_epochs:
                ax.axvline(x=max(bce_epochs), color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
                ax.text(max(bce_epochs), ax.get_ylim()[1] * 0.9, 'Stage Transition', 
                       ha='center', fontsize=10, color='gray')
            
            plt.tight_layout()
            combined_path = training_dir / f'training_{timestamp}_two_stage.png'
            plt.savefig(combined_path, dpi=300, bbox_inches='tight')
            print(f"✓ 2-stage training plot saved: {combined_path}")
            plt.close()
    
    print(f"\n{'='*60}")
    print(f"  All training visualizations saved to: {training_dir}")
    print(f"{'='*60}")


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
    
    # Load test silver labels for self-training
    print("Loading test silver labels...")
    test_labels, test_confidences = SilverLabelGenerator.load_labels(
        args.test_labels_path
    )
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        data_loader.train_corpus,
        data_loader.test_corpus,
        train_labels,
        train_confidences,
        test_labels=test_labels,
        test_confidences=test_confidences,
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
        if len(stats['losses']) > 0:
            print(f"Final loss: {stats['losses'][-1]:.4f}")
        else:
            print(f"Warning: No self-training iterations completed (no confident pseudo-labels)")
        
        # Save combined training history
        training_history['self_training_stats'] = stats
        history_path = Path(args.output_dir) / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Generate visualizations
        print(f"\n=== Generating Visualizations ===")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_curves(training_history, args.training_dir, timestamp)
        
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
        history_path = Path(args.output_dir) / "training_history.json"
        history_path.parent.mkdir(parents=True, exist_ok=True)
        with open(history_path, 'w') as f:
            json.dump(training_history, f, indent=2)
        
        # Generate visualizations
        print(f"\n=== Generating Visualizations ===")
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_training_curves(training_history, args.training_dir, timestamp)
    
    # Save final model with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save with timestamp
    timestamped_model_path = Path(args.output_dir) / f"model_{args.model_type}_{timestamp}.pt"
    torch.save(model.state_dict(), timestamped_model_path)
    print(f"\n✓ Model saved: {timestamped_model_path}")
    
    # Also save as best_model.pt (for backward compatibility)
    best_model_path = Path(args.output_dir) / "best_model.pt"
    torch.save(model.state_dict(), best_model_path)
    print(f"✓ Best model link: {best_model_path}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    
    # Data
    parser.add_argument('--train_labels_path', type=str, 
                       default='data/intermediate/train_silver_labels.pkl')
    parser.add_argument('--test_labels_path', type=str,
                       default='data/intermediate/test_silver_labels.pkl')
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
                       help='Output directory. If not specified, defaults to data/models/{model_type}')
    parser.add_argument('--training_dir', type=str, default=None,
                       help='Training visualization directory. If not specified, defaults to results/training/{model_type}')
    parser.add_argument('--save_every', type=int, default=1)
    
    args = parser.parse_args()
    
    # Set output_dir based on model_type if not specified
    if args.output_dir is None:
        args.output_dir = f'data/models/{args.model_type}'
    
    # Set training_dir based on model_type if not specified
    if args.training_dir is None:
        args.training_dir = f'results/training/{args.model_type}'
    
    # Train
    train_baseline_model(args)


if __name__ == "__main__":
    main()
