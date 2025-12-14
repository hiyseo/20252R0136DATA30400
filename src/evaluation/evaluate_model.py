"""
Model evaluation script for trained models.

Evaluates a trained model on test set and generates comprehensive metrics.
"""

import os
import sys
sys.path.append('.')

import torch
import numpy as np
import argparse
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import DataLoader as DataPreprocessor
from src.dataset import create_dataloaders
from src.models.encoder import TextEncoder, get_device
from src.silver_labeling.generate_silver_labels import SilverLabelGenerator
from src.utils.metrics import (
    compute_multilabel_metrics, 
    find_optimal_threshold,
    top_k_accuracy,
    exact_match_ratio,
    print_metrics
)


@torch.no_grad()
def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on given dataloader.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Device to run evaluation on
        threshold: Threshold for binary predictions
        
    Returns:
        Dictionary with predictions, scores, labels, and metrics
    """
    model.eval()
    
    all_logits = []
    all_labels = []
    
    print("Generating predictions...")
    for batch in tqdm(dataloader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)
        
        all_logits.append(logits.cpu())
        all_labels.append(labels.cpu())
    
    # Concatenate all batches
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    # Convert to numpy
    scores = torch.sigmoid(all_logits).numpy()
    labels = all_labels.numpy()
    
    # Find optimal threshold
    print("\nFinding optimal threshold...")
    optimal_threshold, best_f1 = find_optimal_threshold(labels, scores, metric='micro_f1')
    
    # Use provided threshold or optimal
    eval_threshold = threshold if threshold != 0.5 else optimal_threshold
    
    # Generate predictions
    predictions = (scores >= eval_threshold).astype(int)
    
    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_multilabel_metrics(labels, predictions)
    
    # Additional metrics
    metrics['optimal_threshold'] = optimal_threshold
    metrics['eval_threshold'] = eval_threshold
    metrics['top_3_accuracy'] = top_k_accuracy(labels, scores, k=3)
    metrics['top_5_accuracy'] = top_k_accuracy(labels, scores, k=5)
    metrics['exact_match_ratio'] = exact_match_ratio(labels, predictions)
    
    # Statistics
    metrics['avg_labels_per_sample'] = labels.sum(axis=1).mean()
    metrics['avg_predictions_per_sample'] = predictions.sum(axis=1).mean()
    
    return {
        'predictions': predictions,
        'scores': scores,
        'labels': labels,
        'metrics': metrics
    }


def plot_evaluation_results(results, output_dir, model_type, model_filename):
    """
    Generate evaluation visualizations.
    
    Args:
        results: Evaluation results dictionary
        output_dir: Directory to save plots (results/evaluations/{model_type}/)
        model_type: Model type name
        model_filename: Model filename (stem) for tracking (e.g., 'model_baseline_20250114_153020')
    """
    # Save to results/evaluation/{model_type}/ directory
    eval_dir = Path('results/evaluation') / model_type
    eval_dir.mkdir(parents=True, exist_ok=True)
    
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette('Set2')
    
    predictions = results['predictions']
    scores = results['scores']
    labels = results['labels']
    metrics = results['metrics']
    
    # 1. Prediction confidence distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Positive predictions confidence
    positive_scores = scores[predictions == 1]
    if len(positive_scores) > 0:
        axes[0].hist(positive_scores, bins=50, color='#2ecc71', alpha=0.7, edgecolor='black')
        axes[0].set_xlabel('Confidence Score', fontsize=12)
        axes[0].set_ylabel('Frequency', fontsize=12)
        axes[0].set_title('Positive Predictions Confidence Distribution', fontsize=13, fontweight='bold')
        axes[0].axvline(metrics['eval_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold: {metrics["eval_threshold"]:.2f}')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
    
    # Negative predictions confidence
    negative_scores = scores[predictions == 0]
    if len(negative_scores) > 0:
        axes[1].hist(negative_scores, bins=50, color='#e74c3c', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Confidence Score', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Negative Predictions Confidence Distribution', fontsize=13, fontweight='bold')
        axes[1].axvline(metrics['eval_threshold'], color='red', linestyle='--', linewidth=2, label=f'Threshold: {metrics["eval_threshold"]:.2f}')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    conf_dist_path = eval_dir / f'eval_{model_filename}_confidence_distribution.png'
    plt.savefig(conf_dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence distribution saved: {conf_dist_path}")
    plt.close()
    
    # 2. Labels per sample distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    true_labels_per_sample = labels.sum(axis=1)
    pred_labels_per_sample = predictions.sum(axis=1)
    
    axes[0].hist(true_labels_per_sample, bins=20, color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('Number of Labels', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_title('True Labels per Sample', fontsize=13, fontweight='bold')
    axes[0].axvline(true_labels_per_sample.mean(), color='red', linestyle='--', linewidth=2, 
                   label=f'Mean: {true_labels_per_sample.mean():.2f}')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(pred_labels_per_sample, bins=20, color='#9b59b6', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('Number of Labels', fontsize=12)
    axes[1].set_ylabel('Number of Samples', fontsize=12)
    axes[1].set_title('Predicted Labels per Sample', fontsize=13, fontweight='bold')
    axes[1].axvline(pred_labels_per_sample.mean(), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {pred_labels_per_sample.mean():.2f}')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    labels_dist_path = eval_dir / f'eval_{model_filename}_labels_per_sample_distribution.png'
    plt.savefig(labels_dist_path, dpi=300, bbox_inches='tight')
    print(f"✓ Labels distribution saved: {labels_dist_path}")
    plt.close()
    
    # 3. Metrics bar chart
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metric_names = ['micro_f1', 'macro_f1', 'samples_f1', 'micro_precision', 'macro_precision', 
                   'micro_recall', 'macro_recall', 'top_3_accuracy', 'top_5_accuracy']
    metric_values = [metrics[m] for m in metric_names]
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(metric_names)))
    bars = ax.barh(metric_names, metric_values, color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Score', fontsize=12)
    ax.set_title(f'Evaluation Metrics - {model_type}', fontsize=14, fontweight='bold')
    ax.set_xlim(0, 1.0)
    ax.grid(True, axis='x', alpha=0.3)
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metric_values)):
        ax.text(value + 0.02, i, f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    metrics_chart_path = eval_dir / f'eval_{model_filename}_metrics.png'
    plt.savefig(metrics_chart_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics chart saved: {metrics_chart_path}")
    plt.close()
    
    # 4. F1/Precision/Recall comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['Micro', 'Macro', 'Samples']
    f1_scores = [metrics['micro_f1'], metrics['macro_f1'], metrics['samples_f1']]
    precision_scores = [metrics['micro_precision'], metrics['macro_precision'], 
                       metrics.get('samples_precision', 0)]  # samples_precision may not exist
    recall_scores = [metrics['micro_recall'], metrics['macro_recall'], 
                    metrics.get('samples_recall', 0)]
    
    x = np.arange(len(categories))
    width = 0.25
    
    bars1 = ax.bar(x - width, f1_scores, width, label='F1', color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x, precision_scores, width, label='Precision', color='#2ecc71', alpha=0.8, edgecolor='black')
    bars3 = ax.bar(x + width, recall_scores, width, label='Recall', color='#e74c3c', alpha=0.8, edgecolor='black')
    
    ax.set_xlabel('Averaging Method', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('F1/Precision/Recall Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    comparison_path = eval_dir / f'eval_{model_filename}_f1_precision_recall.png'
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"✓ F1/Precision/Recall comparison saved: {comparison_path}")
    plt.close()
    
    # 5. Top-k Accuracy comparison
    fig, ax = plt.subplots(figsize=(8, 6))
    
    top_k_metrics = {
        'Top-3': metrics.get('top_3_accuracy', 0),
        'Top-5': metrics.get('top_5_accuracy', 0),
        'Exact Match': metrics.get('exact_match_ratio', 0)
    }
    
    colors = ['#9b59b6', '#f39c12', '#e74c3c']
    bars = ax.bar(top_k_metrics.keys(), top_k_metrics.values(), color=colors, alpha=0.8, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Top-k Accuracy Metrics', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(True, axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
               f'{height:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    topk_path = eval_dir / f'eval_{model_filename}_topk_accuracy.png'
    plt.savefig(topk_path, dpi=300, bbox_inches='tight')
    print(f"✓ Top-k accuracy saved: {topk_path}")
    plt.close()
    
    # 6. Per-class performance (top/bottom classes)
    if len(predictions.shape) > 1 and predictions.shape[1] > 0:
        # Calculate per-class F1
        per_class_f1 = []
        for i in range(predictions.shape[1]):
            if labels[:, i].sum() > 0:  # Only for classes with samples
                tp = ((predictions[:, i] == 1) & (labels[:, i] == 1)).sum()
                fp = ((predictions[:, i] == 1) & (labels[:, i] == 0)).sum()
                fn = ((predictions[:, i] == 0) & (labels[:, i] == 1)).sum()
                
                if tp + fp > 0 and tp + fn > 0:
                    prec = tp / (tp + fp)
                    rec = tp / (tp + fn)
                    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
                    per_class_f1.append((i, f1, labels[:, i].sum()))
        
        if len(per_class_f1) > 10:
            # Sort by F1
            per_class_f1.sort(key=lambda x: x[1], reverse=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Top 10 classes
            top_10 = per_class_f1[:10]
            class_ids_top = [f"Class {x[0]}" for x in top_10]
            f1_scores_top = [x[1] for x in top_10]
            
            ax1.barh(class_ids_top, f1_scores_top, color='#2ecc71', alpha=0.8, edgecolor='black')
            ax1.set_xlabel('F1 Score', fontsize=12)
            ax1.set_title('Top 10 Classes by F1 Score', fontsize=13, fontweight='bold')
            ax1.set_xlim(0, 1.0)
            ax1.grid(True, axis='x', alpha=0.3)
            ax1.invert_yaxis()
            
            # Bottom 10 classes
            bottom_10 = per_class_f1[-10:]
            class_ids_bottom = [f"Class {x[0]}" for x in bottom_10]
            f1_scores_bottom = [x[1] for x in bottom_10]
            
            ax2.barh(class_ids_bottom, f1_scores_bottom, color='#e74c3c', alpha=0.8, edgecolor='black')
            ax2.set_xlabel('F1 Score', fontsize=12)
            ax2.set_title('Bottom 10 Classes by F1 Score', fontsize=13, fontweight='bold')
            ax2.set_xlim(0, 1.0)
            ax2.grid(True, axis='x', alpha=0.3)
            ax2.invert_yaxis()
            
            plt.tight_layout()
            perclass_path = eval_dir / f'eval_{model_filename}_per_class_performance.png'
            plt.savefig(perclass_path, dpi=300, bbox_inches='tight')
            print(f"✓ Per-class performance saved: {perclass_path}")
            plt.close()
    
    print(f"\n{'='*60}")
    print(f"  All evaluation visualizations saved to: {eval_dir}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    
    # Model
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.pt file)')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased',
                       help='Base model name (must match training)')
    parser.add_argument('--model_type', type=str, default='baseline',
                       help='Model type for directory naming')
    
    # Data
    parser.add_argument('--test_labels_path', type=str,
                       default='data/intermediate/test_silver_labels.pkl',
                       help='Path to test labels (silver labels)')
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Evaluation
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Threshold for binary predictions (0.5 = auto-find optimal)')
    
    # Output
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to file')
    
    args = parser.parse_args()
    
    # Set output_dir
    if args.output_dir is None:
        args.output_dir = f'results/evaluations/{args.model_type}'
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load data
    print("\n=== Loading Data ===")
    data_loader = DataPreprocessor()
    data_loader.load_all()
    
    # Load test labels
    print("Loading test labels...")
    test_labels, test_confidences = SilverLabelGenerator.load_labels(
        args.test_labels_path
    )
    
    # Create test dataloader
    print("Creating test dataloader...")
    from src.dataset import MultiLabelDataset
    from torch.utils.data import DataLoader
    from transformers import AutoTokenizer
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    test_dataset = MultiLabelDataset(
        texts=data_loader.test_corpus,
        labels=test_labels,
        confidences=test_confidences,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    # Load model
    print(f"\n=== Loading Model: {args.model_path} ===")
    model = TextEncoder(
        model_name=args.model_name,
        num_classes=len(data_loader.classes),
        dropout=0.1
    ).to(device)
    
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("✓ Model loaded successfully")
    
    # Evaluate
    print(f"\n=== Evaluation ===")
    results = evaluate_model(model, test_loader, device, threshold=args.threshold)
    
    # Print metrics
    print("\n" + "="*60)
    print("  EVALUATION RESULTS")
    print("="*60)
    print_metrics(results['metrics'])
    
    # Save metrics
    metrics_path = Path(args.output_dir) / 'evaluation_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"\n✓ Metrics saved: {metrics_path}")
    
    # Save predictions
    if args.save_predictions:
        predictions_path = Path(args.output_dir) / 'predictions.npz'
        np.savez(
            predictions_path,
            predictions=results['predictions'],
            scores=results['scores'],
            labels=results['labels']
        )
        print(f"✓ Predictions saved: {predictions_path}")
    
    # Generate visualizations
    print(f"\n=== Generating Visualizations ===")
    # Extract model filename (without .pt extension) for tracking
    model_filename = Path(args.model_path).stem  # e.g., 'model_baseline_20250114_153020' or 'best_model'
    plot_evaluation_results(results, args.output_dir, args.model_type, model_filename)
    
    # Also save evaluation metrics JSON with model filename
    metrics_json_path = Path(args.output_dir) / f'eval_{model_filename}_metrics.json'
    with open(metrics_json_path, 'w') as f:
        json.dump(results['metrics'], f, indent=2)
    print(f"✓ Metrics JSON saved: {metrics_json_path}")
    
    print("\n✓ Evaluation completed successfully!")


if __name__ == "__main__":
    main()
