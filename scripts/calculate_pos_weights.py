"""
Calculate positive weights for imbalanced dataset.
"""

import pickle
import numpy as np
import torch

def calculate_pos_weights(label_path, output_path):
    """Calculate positive weights based on class frequency."""
    
    with open(label_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    
    # Count positive samples per class
    pos_counts = labels.sum(axis=0)
    neg_counts = len(labels) - pos_counts
    
    # Avoid division by zero
    pos_counts = np.maximum(pos_counts, 1)
    
    # Positive weight = neg_count / pos_count
    pos_weights = neg_counts / pos_counts
    
    # Clip extreme values
    pos_weights = np.clip(pos_weights, 1.0, 100.0)
    
    print(f"Pos weights - min: {pos_weights.min():.2f}, max: {pos_weights.max():.2f}, mean: {pos_weights.mean():.2f}")
    
    # Save as tensor
    pos_weights_tensor = torch.FloatTensor(pos_weights)
    torch.save(pos_weights_tensor, output_path)
    print(f"Saved to: {output_path}")
    
    return pos_weights


if __name__ == "__main__":
    weights = calculate_pos_weights(
        'data/intermediate/train_silver_labels.pkl',
        'data/intermediate/pos_weights.pt'
    )
