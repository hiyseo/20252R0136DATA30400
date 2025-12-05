"""
Check silver label generation results.
"""

import pickle
import numpy as np

# Load silver labels
print("=== Loading Silver Labels ===")
with open("data/intermediate/train_silver_labels.pkl", "rb") as f:
    train_data = pickle.load(f)

with open("data/intermediate/test_silver_labels.pkl", "rb") as f:
    test_data = pickle.load(f)

train_labels = train_data['labels']
train_confidences = train_data['confidences']
test_labels = test_data['labels']
test_confidences = test_data['confidences']

print(f"\n✓ Loaded train labels: {train_labels.shape}")
print(f"✓ Loaded test labels: {test_labels.shape}")

# Training data statistics
print("\n=== Training Data Statistics ===")
labeled_samples = np.sum(np.sum(train_labels, axis=1) > 0)
total_samples = train_labels.shape[0]
coverage = labeled_samples / total_samples * 100

print(f"Total samples: {total_samples}")
print(f"Labeled samples: {labeled_samples} ({coverage:.2f}%)")
print(f"Unlabeled samples: {total_samples - labeled_samples}")

# Labels per sample
labels_per_sample = np.sum(train_labels, axis=1)
print(f"\nLabels per sample:")
print(f"  Mean: {np.mean(labels_per_sample[labels_per_sample > 0]):.2f}")
print(f"  Median: {np.median(labels_per_sample[labels_per_sample > 0]):.1f}")
print(f"  Min: {np.min(labels_per_sample[labels_per_sample > 0])}")
print(f"  Max: {np.max(labels_per_sample)}")

# Class distribution
samples_per_class = np.sum(train_labels, axis=0)
print(f"\nSamples per class:")
print(f"  Mean: {np.mean(samples_per_class[samples_per_class > 0]):.2f}")
print(f"  Median: {np.median(samples_per_class[samples_per_class > 0]):.1f}")
print(f"  Min: {np.min(samples_per_class[samples_per_class > 0])}")
print(f"  Max: {np.max(samples_per_class)}")
print(f"  Classes with labels: {np.sum(samples_per_class > 0)}/531")

# Test data statistics
print("\n=== Test Data Statistics ===")
test_labeled_samples = np.sum(np.sum(test_labels, axis=1) > 0)
test_total_samples = test_labels.shape[0]
test_coverage = test_labeled_samples / test_total_samples * 100

print(f"Total samples: {test_total_samples}")
print(f"Labeled samples: {test_labeled_samples} ({test_coverage:.2f}%)")
print(f"Unlabeled samples: {test_total_samples - test_labeled_samples}")

test_labels_per_sample = np.sum(test_labels, axis=1)
print(f"\nLabels per sample:")
print(f"  Mean: {np.mean(test_labels_per_sample[test_labels_per_sample > 0]):.2f}")
print(f"  Median: {np.median(test_labels_per_sample[test_labels_per_sample > 0]):.1f}")

# Confidence statistics
print("\n=== Confidence Statistics ===")
train_conf_values = train_confidences[train_confidences > 0]
print(f"Training confidence:")
print(f"  Mean: {np.mean(train_conf_values):.3f}")
print(f"  Median: {np.median(train_conf_values):.3f}")
print(f"  Std: {np.std(train_conf_values):.3f}")

# Sample some labeled examples
print("\n=== Sample Labeled Examples ===")
for i in range(5):
    if np.sum(train_labels[i]) > 0:
        num_labels = int(np.sum(train_labels[i]))
        label_ids = np.where(train_labels[i] == 1)[0]
        confidences = train_confidences[i][label_ids]
        print(f"\nSample {i}: {num_labels} labels")
        print(f"  Class IDs: {label_ids[:5]}...")
        print(f"  Confidences: {confidences[:5]}...")
