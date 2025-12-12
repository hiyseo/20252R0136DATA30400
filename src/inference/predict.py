"""
Inference script for generating predictions on test set.
Supports BERT, GCN, and GAT models.
"""

import os
import sys
import argparse
import pickle
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import HierarchicalDataLoader
from src.models.encoder import BERTEncoder
from src.models.classifier import MultiLabelClassifier
from src.models.gnn_classifier import GCNClassifier, GATClassifier, build_adjacency_matrix
from src.utils.logger import setup_logger

logger = setup_logger("Inference")


def load_model(checkpoint_path: str, data_loader: HierarchicalDataLoader, device: torch.device):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth file)
        data_loader: Data loader with class information
        device: Device to load model on
        
    Returns:
        model: Loaded model
        model_type: Type of model (bert/gcn/gat)
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'bert')
    num_classes = data_loader.num_classes
    
    logger.info(f"Loading {model_type.upper()} model from {checkpoint_path}")
    
    # Initialize encoder
    model_name = config.get('model_name', 'bert-base-uncased')
    dropout = config.get('dropout', 0.1)
    encoder = BERTEncoder(model_name=model_name, dropout=dropout)
    
    # Initialize classifier based on type
    if model_type == 'gcn':
        # Build adjacency matrix
        hierarchy_graph = data_loader.load_hierarchy()
        adj_matrix = build_adjacency_matrix(hierarchy_graph, num_classes, self_loop=True)
        
        model = GCNClassifier(
            text_encoder=encoder,
            num_classes=num_classes,
            hidden_dim=config.get('gnn_hidden_dim', 512),
            num_gcn_layers=config.get('gnn_num_layers', 2),
            dropout=dropout,
            adjacency_matrix=adj_matrix
        )
    elif model_type == 'gat':
        # Build adjacency matrix
        hierarchy_graph = data_loader.load_hierarchy()
        adj_matrix = build_adjacency_matrix(hierarchy_graph, num_classes, self_loop=True)
        
        model = GATClassifier(
            text_encoder=encoder,
            num_classes=num_classes,
            hidden_dim=config.get('gnn_hidden_dim', 512),
            num_gat_layers=config.get('gnn_num_layers', 2),
            num_heads=config.get('gnn_num_heads', 4),
            dropout=dropout,
            adjacency_matrix=adj_matrix
        )
    else:  # bert
        model = MultiLabelClassifier(encoder=encoder, num_classes=num_classes)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully")
    logger.info(f"Trained for {checkpoint.get('epoch', 'unknown')} epochs")
    logger.info(f"Best validation metric: {checkpoint.get('best_metric', 'unknown')}")
    
    return model, model_type


@torch.no_grad()
def predict(model, data_loader, device: torch.device, threshold: float = 0.5):
    """
    Generate predictions on dataset.
    
    Args:
        model: Trained model
        data_loader: DataLoader for test set
        device: Device to run inference on
        threshold: Threshold for binary classification
        
    Returns:
        all_pids: List of product IDs
        all_predictions: List of predicted class indices (multi-label)
        all_probs: List of prediction probabilities
    """
    model.eval()
    
    all_pids = []
    all_predictions = []
    all_probs = []
    
    logger.info(f"Running inference with threshold={threshold}")
    
    for batch in tqdm(data_loader, desc="Predicting"):
        pids = batch['pid']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        
        # Apply threshold
        predictions = (probs >= threshold).long()
        
        # Convert to list of class indices
        for i in range(len(pids)):
            pid = pids[i]
            pred_classes = torch.where(predictions[i] == 1)[0].cpu().numpy().tolist()
            prob_values = probs[i].cpu().numpy()
            
            # If no predictions, take top-1
            if len(pred_classes) == 0:
                pred_classes = [torch.argmax(probs[i]).item()]
            
            all_pids.append(pid)
            all_predictions.append(pred_classes)
            all_probs.append(prob_values)
    
    logger.info(f"Generated predictions for {len(all_pids)} samples")
    avg_labels = np.mean([len(p) for p in all_predictions])
    logger.info(f"Average labels per sample: {avg_labels:.2f}")
    
    return all_pids, all_predictions, all_probs


def main():
    parser = argparse.ArgumentParser(description="Generate predictions on test set")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/raw/Amazon_products',
                       help='Path to data directory')
    parser.add_argument('--output_path', type=str, default='predictions/test_predictions.pkl',
                       help='Path to save predictions')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary classification')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for inference')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/mps/cpu). Auto-detect if not specified')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
    
    logger.info(f"Using device: {device}")
    
    # Load data
    logger.info(f"Loading test data from {args.data_dir}")
    data_loader_obj = HierarchicalDataLoader(data_dir=args.data_dir)
    
    # Get test data
    test_dataset = data_loader_obj.load_test_data()
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )
    
    logger.info(f"Test set size: {len(test_dataset)}")
    
    # Load model
    model, model_type = load_model(args.model_path, data_loader_obj, device)
    
    # Generate predictions
    pids, predictions, probs = predict(model, test_loader, device, args.threshold)
    
    # Save predictions
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    results = {
        'pids': pids,
        'predictions': predictions,
        'probabilities': probs,
        'model_type': model_type,
        'threshold': args.threshold,
        'model_path': args.model_path
    }
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"Predictions saved to {output_path}")
    
    # Print statistics
    logger.info("\n=== Prediction Statistics ===")
    logger.info(f"Total samples: {len(pids)}")
    logger.info(f"Average labels per sample: {np.mean([len(p) for p in predictions]):.2f}")
    logger.info(f"Min labels: {min([len(p) for p in predictions])}")
    logger.info(f"Max labels: {max([len(p) for p in predictions])}")
    
    # Label distribution
    label_counts = {}
    for pred in predictions:
        for label in pred:
            label_counts[label] = label_counts.get(label, 0) + 1
    
    logger.info(f"Unique predicted classes: {len(label_counts)}")
    top_classes = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info(f"Top 10 predicted classes: {top_classes}")


if __name__ == '__main__':
    main()