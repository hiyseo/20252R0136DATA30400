"""
Rescale silver label confidences to higher range.
Original: 0.1~0.5 → New: 0.5~0.9
"""

import pickle
import numpy as np

def rescale_confidences(input_path, output_path, 
                       old_min=0.1, old_max=0.5,
                       new_min=0.5, new_max=0.9):
    """
    Rescale confidence values to higher range.
    
    Args:
        input_path: Path to original silver labels
        output_path: Path to save rescaled labels
        old_min, old_max: Original confidence range
        new_min, new_max: Target confidence range
    """
    
    print(f"Loading from: {input_path}")
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    confidences = data['confidences']
    
    # Get positive confidences
    positive_mask = labels > 0
    positive_confs = confidences[positive_mask]
    
    print(f"\nOriginal confidence:")
    print(f"  Min: {positive_confs.min():.4f}")
    print(f"  Max: {positive_confs.max():.4f}")
    print(f"  Mean: {positive_confs.mean():.4f}")
    
    # Rescale: (x - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    confidences_rescaled = confidences.copy()
    
    # Only rescale positive confidences
    positive_confs_rescaled = (positive_confs - old_min) / (old_max - old_min) * (new_max - new_min) + new_min
    positive_confs_rescaled = np.clip(positive_confs_rescaled, new_min, new_max)
    
    confidences_rescaled[positive_mask] = positive_confs_rescaled
    
    print(f"\nRescaled confidence:")
    print(f"  Min: {confidences_rescaled[positive_mask].min():.4f}")
    print(f"  Max: {confidences_rescaled[positive_mask].max():.4f}")
    print(f"  Mean: {confidences_rescaled[positive_mask].mean():.4f}")
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump({
            'labels': labels,
            'confidences': confidences_rescaled
        }, f)
    
    print(f"\nSaved to: {output_path}")
    
    return labels, confidences_rescaled


if __name__ == "__main__":
    import sys
    import os
    
    # Check if already rescaled
    train_path = 'data/intermediate/train_silver_labels.pkl'
    
    # Quick check: load and see if already rescaled
    with open(train_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    confidences = data['confidences']
    positive_mask = labels > 0
    
    if len(confidences[positive_mask]) > 0:
        current_mean = confidences[positive_mask].mean()
        
        if current_mean > 0.4:
            print("\n" + "="*60)
            print("⚠️  WARNING: Labels appear to be already rescaled!")
            print("="*60)
            print(f"Current mean confidence: {current_mean:.4f}")
            print("\nIf you want to rescale again, delete the current labels and regenerate.")
            print("Skipping rescaling to prevent double-scaling.")
            sys.exit(0)
    
    # Rescale train labels
    print("="*60)
    print("TRAIN LABELS")
    print("="*60)
    rescale_confidences(
        'data/intermediate/train_silver_labels.pkl',
        'data/intermediate/train_silver_labels.pkl',  # Overwrite
        new_min=0.5, new_max=0.9
    )
    
    print("\n" + "="*60)
    print("TEST LABELS")
    print("="*60)
    rescale_confidences(
        'data/intermediate/test_silver_labels.pkl',
        'data/intermediate/test_silver_labels.pkl',  # Overwrite
        new_min=0.5, new_max=0.9
    )
    
    print("\n✅ Rescaling complete!")
    print("\n⚠️  Note: Do NOT run this script again unless you regenerate silver labels.")
