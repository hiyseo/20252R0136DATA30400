"""
Evaluation metrics for multi-label classification.
"""

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, hamming_loss
from typing import Tuple, Dict


def compute_multilabel_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive metrics for multi-label classification.
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        y_pred: Predicted binary labels (n_samples, n_classes)
        
    Returns:
        Dictionary of metric names and values
    """
    metrics = {
        'micro_f1': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'samples_f1': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'micro_precision': precision_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'micro_recall': recall_score(y_true, y_pred, average='micro', zero_division=0),
        'macro_recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'hamming_loss': hamming_loss(y_true, y_pred),
    }
    
    return metrics


def find_optimal_threshold(y_true: np.ndarray, y_scores: np.ndarray, 
                          metric: str = 'micro_f1') -> Tuple[float, float]:
    """
    Find optimal threshold for converting scores to binary predictions.
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        y_scores: Predicted scores (n_samples, n_classes)
        metric: Metric to optimize ('micro_f1', 'macro_f1', or 'samples_f1')
        
    Returns:
        Tuple of (best_threshold, best_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_scores >= threshold).astype(int)
        metrics = compute_multilabel_metrics(y_true, y_pred)
        
        if metrics[metric] > best_score:
            best_score = metrics[metric]
            best_threshold = threshold
    
    return best_threshold, best_score


def top_k_accuracy(y_true: np.ndarray, y_scores: np.ndarray, k: int = 3) -> float:
    """
    Compute top-k accuracy for multi-label classification.
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        y_scores: Predicted scores (n_samples, n_classes)
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    n_samples = y_true.shape[0]
    correct = 0
    
    for i in range(n_samples):
        # Get top-k predictions
        top_k_idx = np.argsort(y_scores[i])[-k:]
        # Check if any true label is in top-k
        if np.any(y_true[i][top_k_idx] == 1):
            correct += 1
    
    return correct / n_samples


def exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute exact match ratio (all labels must match).
    
    Args:
        y_true: True binary labels (n_samples, n_classes)
        y_pred: Predicted binary labels (n_samples, n_classes)
        
    Returns:
        Exact match ratio
    """
    return np.mean(np.all(y_true == y_pred, axis=1))


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """Pretty print metrics."""
    if prefix:
        print(f"\n{prefix}")
    print("-" * 50)
    for metric_name, value in metrics.items():
        print(f"{metric_name:20s}: {value:.4f}")
    print("-" * 50)