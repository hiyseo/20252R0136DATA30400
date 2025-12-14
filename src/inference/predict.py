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
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.data_preprocessing import DataLoader
from src.dataset import ProductReviewDataset
from src.models.encoder import TextEncoder
from src.models.gnn_classifier import GCNClassifier, GATClassifier, build_adjacency_matrix
from src.utils.logger import setup_logger

logger = setup_logger("Inference")


def load_model(checkpoint_path: str, num_classes: int, hierarchy_graph=None, device: torch.device = None):
    """
    Load trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pth or .pt file)
        num_classes: Number of classes
        hierarchy_graph: Hierarchy graph (for GNN models)
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Get model configuration
    config = checkpoint.get('config', {})
    if 'model' in config:
        model_config = config['model']
        model_type = model_config.get('model_type', 'bert')
        model_name = model_config.get('model_name', 'bert-base-uncased')
        dropout = model_config.get('dropout', 0.1)
    else:
        # Fallback for older checkpoints
        model_type = config.get('model_type', 'bert')
        model_name = config.get('model_name', 'bert-base-uncased')
        dropout = config.get('dropout', 0.1)
    
    logger.info(f"Model type: {model_type.upper()}")
    
    # Initialize model based on type
    if model_type == 'gcn':
        if hierarchy_graph is None:
            raise ValueError("Hierarchy graph required for GCN model")
        
        adj_matrix = build_adjacency_matrix(hierarchy_graph, num_classes, self_loop=True)
        encoder = TextEncoder(model_name=model_name, num_classes=num_classes, dropout=dropout)
        
        model = GCNClassifier(
            text_encoder=encoder,
            num_classes=num_classes,
            hidden_dim=config.get('model', {}).get('gnn_hidden_dim', 512),
            num_gcn_layers=config.get('model', {}).get('gnn_num_layers', 2),
            dropout=dropout,
            adjacency_matrix=adj_matrix
        )
    elif model_type == 'gat':
        if hierarchy_graph is None:
            raise ValueError("Hierarchy graph required for GAT model")
        
        adj_matrix = build_adjacency_matrix(hierarchy_graph, num_classes, self_loop=True)
        encoder = TextEncoder(model_name=model_name, num_classes=num_classes, dropout=dropout)
        
        model = GATClassifier(
            text_encoder=encoder,
            num_classes=num_classes,
            hidden_dim=config.get('model', {}).get('gnn_hidden_dim', 512),
            num_gat_layers=config.get('model', {}).get('gnn_num_layers', 2),
            num_heads=config.get('model', {}).get('gnn_num_heads', 4),
            dropout=dropout,
            adjacency_matrix=adj_matrix
        )
    else:  # bert
        model = TextEncoder(model_name=model_name, num_classes=num_classes, dropout=dropout)
    
    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"✓ Model loaded successfully")
    if 'epoch' in checkpoint:
        logger.info(f"  Trained for {checkpoint['epoch']} epochs")
    
    return model, config


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
        doc_ids = batch['doc_id']
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        probs = torch.sigmoid(logits)
        
        # Apply threshold
        predictions = (probs >= threshold).long()
        
        # Convert to list of class indices
        for i in range(len(doc_ids)):
            doc_id = doc_ids[i].item() if torch.is_tensor(doc_ids[i]) else doc_ids[i]
            pred_classes = torch.where(predictions[i] == 1)[0].cpu().numpy().tolist()
            prob_values = probs[i].cpu().numpy()
            
            # If no predictions, take top-1
            if len(pred_classes) == 0:
                pred_classes = [torch.argmax(probs[i]).item()]
            
            all_pids.append(doc_id)
            all_predictions.append(pred_classes)
            all_probs.append(prob_values)
def main():
    parser = argparse.ArgumentParser(description="Generate predictions on test set")
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--data_dir', type=str, default='data/raw/Amazon_products',
                       help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='results/predictions',
                       help='Directory to save predictions')
    parser.add_argument('--model_name', type=str, default=None,
                       help='Model name for file naming (e.g., baseline, gcn). Auto-detect if not specified')
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
    logger.info(f"Loading data from {args.data_dir}")
    data_loader_obj = DataLoader(data_dir=args.data_dir)
    data_loader_obj.load_all()
    
    num_classes = data_loader_obj.num_classes
    hierarchy_graph = data_loader_obj.hierarchy
    
    logger.info(f"Number of classes: {num_classes}")
    
    # Create test dataset
    test_dataset = ProductReviewDataset(
        corpus=data_loader_obj.test_corpus,
        labels=None,  # No labels for test set
        tokenizer_name='bert-base-uncased',
        max_length=128
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 to avoid multiprocessing issues
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # Load model
    model, config = load_model(args.model_path, num_classes, hierarchy_graph, device)
    
    # Determine model name
    if args.model_name:
        model_name = args.model_name
    else:
        # Auto-detect from config or path
        if 'model' in config and 'model_type' in config['model']:
            model_name = config['model']['model_type']
        else:
            model_name = Path(args.model_path).parent.name
    
    # Generate predictions
    pids, predictions, probs = predict(model, test_loader, device, args.threshold)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract model filename for tracking
    model_filename = Path(args.model_path).stem  # e.g., 'model_baseline_20250114_153020' or 'best_model'
    
    # Save predictions as PKL
    pkl_filename = f"pred_{model_filename}.pkl"
    pkl_path = output_dir / pkl_filename
    
    results = {
        'pids': pids,
        'predictions': predictions,
        'probabilities': probs,
        'config': config,
        'threshold': args.threshold,
        'model_path': args.model_path,
        'model_name': model_name,
        'model_filename': model_filename
    }
    
    with open(pkl_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"✓ Predictions saved to {pkl_path}")
    
    # Save as CSV (Kaggle format)
    csv_filename = f"pred_{model_filename}.csv"
    csv_path = output_dir / csv_filename
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        for pred_classes in predictions:
            class_str = ' '.join(map(str, sorted(pred_classes)))
            f.write(f"{class_str}\n")
    
    logger.info(f"✓ Submission CSV saved to {csv_path}")
    
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