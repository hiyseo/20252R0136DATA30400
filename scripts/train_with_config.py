"""
AWS SageMaker training script with config support.
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
sys.path.append('.')

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
import numpy as np
from tqdm import tqdm

from src.data_preprocessing import DataLoader as DataPreprocessor
from src.dataset import create_dataloaders
from src.models.encoder import TextEncoder, get_device
from src.training.loss_functions import get_loss_function
from src.utils.metrics import compute_multilabel_metrics, find_optimal_threshold, print_metrics
from src.utils.seed import set_seed
from src.utils.logger import setup_logger
from src.silver_labeling.generate_silver_labels import SilverLabelGenerator


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, logger):
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
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    avg_loss = total_loss / num_batches
    logger.info(f"Average training loss: {avg_loss:.4f}")
    
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_name', type=str, help='Override model name')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--num_epochs', type=int, help='Override num epochs')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override with command line arguments
    if args.model_name:
        config['model']['model_name'] = args.model_name
    if args.batch_size:
        config['training']['batch_size'] = args.batch_size
    if args.num_epochs:
        config['training']['num_epochs'] = args.num_epochs
    if args.learning_rate:
        config['training']['learning_rate'] = args.learning_rate
    
    # Setup logger
    logger = setup_logger(
        name="Training",
        log_dir=config['output']['log_dir']
    )
    
    logger.info("="*60)
    logger.info("Starting Training Pipeline")
    logger.info("="*60)
    
    # Set seed
    set_seed(config['misc']['seed'])
    
    # Get device
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info("\n=== Loading Data ===")
    data_loader = DataPreprocessor(data_dir=config['data']['data_dir'])
    data_loader.load_all()
    
    # Load silver labels
    logger.info("Loading silver labels...")
    train_labels, train_confidences = SilverLabelGenerator.load_labels(
        config['data']['train_labels_path']
    )
    
    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, test_loader = create_dataloaders(
        data_loader.train_corpus,
        data_loader.test_corpus,
        train_labels,
        train_confidences,
        tokenizer_name=config['model']['model_name'],
        batch_size=config['training']['batch_size'],
        max_length=config['data']['max_length'],
        num_workers=config['data']['num_workers']
    )
    
    logger.info(f"Train samples: {len(train_loader.dataset)}")
    logger.info(f"Train batches: {len(train_loader)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    logger.info(f"\n=== Creating Model: {config['model']['model_name']} ===")
    model = TextEncoder(
        model_name=config['model']['model_name'],
        num_classes=config['model']['num_classes'],
        dropout=config['model']['dropout'],
        freeze_encoder=config['model']['freeze_encoder']
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function
    criterion = get_loss_function(config['training']['loss_type'])
    logger.info(f"Loss function: {config['training']['loss_type']}")
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Scheduler
    total_steps = len(train_loader) * config['training']['num_epochs']
    warmup_steps = int(total_steps * config['training']['warmup_ratio'])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )
    
    logger.info(f"Total training steps: {total_steps}")
    logger.info(f"Warmup steps: {warmup_steps}")
    
    # Training loop
    logger.info(f"\n=== Training for {config['training']['num_epochs']} epochs ===")
    
    output_dir = Path(config['output']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    best_loss = float('inf')
    
    for epoch in range(config['training']['num_epochs']):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{config['training']['num_epochs']}")
        logger.info(f"{'='*60}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, logger
        )
        
        # Save checkpoint
        if (epoch + 1) % config['evaluation']['save_every'] == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'config': config
            }, checkpoint_path)
            
            logger.info(f"✓ Checkpoint saved: {checkpoint_path}")
            
            # Save best model
            if train_loss < best_loss:
                best_loss = train_loss
                best_model_path = output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"✓ Best model saved: {best_model_path}")
    
    # Save final model
    final_model_path = output_dir / "final_model.pt"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"\n✓ Final model saved: {final_model_path}")
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()
