"""
Quick test script to verify training pipeline works.
Tests with minimal data and 1 epoch.
"""

import sys
sys.path.append('.')

import torch
from src.data_preprocessing import DataLoader
from src.dataset import create_dataloaders
from src.models.encoder import TextEncoder, get_device
from src.training.loss_functions import get_loss_function
from src.silver_labeling.generate_silver_labels import SilverLabelGenerator
from src.utils.seed import set_seed

def quick_test():
    """Quick test of training pipeline."""
    
    print("=== Quick Training Pipeline Test ===\n")
    print("Step 1/7: Setting seed...")
    
    # Set seed
    set_seed(42)
    
    # Device
    print("Step 2/7: Detecting device...")
    device = get_device()
    print(f"Device: {device}\n")
    
    # Load data (use small subset)
    print("Step 3/7: Loading data...")
    data_loader = DataLoader()
    data_loader.load_all()
    
    # Use only first 100 samples for quick test
    train_corpus = data_loader.train_corpus[:100]
    test_corpus = data_loader.test_corpus[:50]
    
    print("Step 4/7: Loading silver labels...")
    train_labels, train_confidences = SilverLabelGenerator.load_labels(
        "data/intermediate/train_silver_labels.pkl"
    )
    train_labels = train_labels[:100]
    train_confidences = train_confidences[:100]
    
    # Create dataloaders
    print("Step 5/7: Creating dataloaders and downloading tokenizer (this may take a while)...")
    train_loader, test_loader = create_dataloaders(
        train_corpus,
        test_corpus,
        train_labels,
        train_confidences,
        tokenizer_name="bert-base-uncased",
        batch_size=8,
        max_length=64,  # Shorter for quick test
        num_workers=0  # No multiprocessing for test
    )
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}\n")
    
    # Create model
    print("Step 6/7: Creating model and downloading BERT (this will take 2-3 minutes on first run)...")
    model = TextEncoder(
        model_name="bert-base-uncased",
        num_classes=531,
        dropout=0.1
    ).to(device)
    print("✓ Model loaded!")
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}\n")
    
    # Loss and optimizer
    criterion = get_loss_function('bce')
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    
    # Test one training step
    print("\nStep 7/7: Testing training step...")
    model.train()
    batch = next(iter(train_loader))
    
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    labels = batch['labels'].to(device)
    
    print(f"Batch shapes:")
    print(f"  input_ids: {input_ids.shape}")
    print(f"  attention_mask: {attention_mask.shape}")
    print(f"  labels: {labels.shape}")
    
    # Forward
    logits = model(input_ids, attention_mask)
    print(f"  logits: {logits.shape}")
    
    loss = criterion(logits, labels)
    print(f"  loss: {loss.item():.4f}")
    
    # Backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print("\n✓ Training step successful!")
    
    # Test inference
    print("\nTesting inference...")
    model.eval()
    test_batch = next(iter(test_loader))
    
    with torch.no_grad():
        test_input_ids = test_batch['input_ids'].to(device)
        test_attention_mask = test_batch['attention_mask'].to(device)
        test_logits = model(test_input_ids, test_attention_mask)
        test_probs = torch.sigmoid(test_logits)
    
    print(f"Test batch size: {test_input_ids.shape[0]}")
    print(f"Predictions shape: {test_probs.shape}")
    print(f"Sample prediction (first 5 classes): {test_probs[0, :5].cpu().numpy()}")
    
    print("\n✓ Inference successful!")
    print("\n" + "="*50)
    print("✅ ALL TESTS PASSED!")
    print("="*50)
    print("\nPipeline is ready for full training on AWS SageMaker.")


if __name__ == "__main__":
    quick_test()
