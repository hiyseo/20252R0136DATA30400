"""
Debug script to analyze silver label quality.
"""

import pickle
import numpy as np
from pathlib import Path

def analyze_silver_labels(label_path):
    """Analyze silver labels statistics."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {label_path}")
    print(f"{'='*60}")
    
    with open(label_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    confidences = data['confidences']
    
    print(f"\nShape: {labels.shape}")
    print(f"Total samples: {labels.shape[0]}")
    print(f"Total classes: {labels.shape[1]}")
    
    # Label statistics
    positive_per_sample = labels.sum(axis=1)
    print(f"\n--- Label Statistics ---")
    print(f"Samples with labels: {(positive_per_sample > 0).sum()} / {len(labels)}")
    print(f"Labels per sample - min: {positive_per_sample.min():.0f}, max: {positive_per_sample.max():.0f}, mean: {positive_per_sample.mean():.2f}")
    
    # Class distribution
    positive_per_class = labels.sum(axis=0)
    print(f"\nLabels per class - min: {positive_per_class.min():.0f}, max: {positive_per_class.max():.0f}, mean: {positive_per_class.mean():.2f}")
    print(f"Classes with no labels: {(positive_per_class == 0).sum()} / {labels.shape[1]}")
    
    # Confidence statistics
    print(f"\n--- Confidence Statistics ---")
    positive_confidences = confidences[labels > 0]
    if len(positive_confidences) > 0:
        print(f"Positive confidence - min: {positive_confidences.min():.4f}, max: {positive_confidences.max():.4f}, mean: {positive_confidences.mean():.4f}")
    else:
        print(f"No positive labels found!")
    
    negative_confidences = confidences[labels == 0]
    print(f"Negative confidence - min: {negative_confidences.min():.4f}, max: {negative_confidences.max():.4f}, mean: {negative_confidences.mean():.4f}")
    
    # Sample analysis
    print(f"\n--- Sample Analysis (first 10) ---")
    for i in range(min(10, len(labels))):
        n_labels = positive_per_sample[i]
        if n_labels > 0:
            sample_confs = confidences[i][labels[i] > 0]
            print(f"Sample {i}: {n_labels:.0f} labels, conf range: {sample_confs.min():.3f}-{sample_confs.max():.3f}")
        else:
            print(f"Sample {i}: 0 labels")
    
    return labels, confidences


if __name__ == "__main__":
    # Analyze train labels
    train_labels, train_confs = analyze_silver_labels('data/intermediate/train_silver_labels.pkl')
    
    # Analyze test labels
    test_labels, test_confs = analyze_silver_labels('data/intermediate/test_silver_labels.pkl')
    
    # Comparison
    print(f"\n{'='*60}")
    print("COMPARISON")
    print(f"{'='*60}")
    print(f"Train avg labels/sample: {train_labels.sum(axis=1).mean():.2f}")
    print(f"Test avg labels/sample:  {test_labels.sum(axis=1).mean():.2f}")
    print(f"\nTrain avg positive conf: {train_confs[train_labels > 0].mean():.4f}")
    
    test_pos_confs = test_confs[test_labels > 0]
    if len(test_pos_confs) > 0:
        print(f"Test avg positive conf:  {test_pos_confs.mean():.4f}")
    else:
        print(f"Test avg positive conf:  NO POSITIVE LABELS!")
