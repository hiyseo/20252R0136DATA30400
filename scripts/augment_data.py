"""
Augment training data by upsampling minority classes.
"""

import pickle
import numpy as np
from collections import Counter

def augment_silver_labels(input_path, output_path, target_min_samples=100):
    """Upsample classes with few samples."""
    
    with open(input_path, 'rb') as f:
        data = pickle.load(f)
    
    labels = data['labels']
    confidences = data['confidences']
    
    # Find minority classes
    pos_counts = labels.sum(axis=0)
    minority_classes = np.where(pos_counts < target_min_samples)[0]
    
    print(f"Found {len(minority_classes)} minority classes")
    
    # Upsample minority class samples
    augmented_labels = [labels]
    augmented_confs = [confidences]
    
    for class_id in minority_classes:
        # Find samples with this class
        has_class = labels[:, class_id] > 0
        class_samples = np.where(has_class)[0]
        
        if len(class_samples) == 0:
            continue
        
        # How many copies needed?
        n_copies = (target_min_samples // len(class_samples)) - 1
        
        if n_copies > 0:
            for _ in range(n_copies):
                augmented_labels.append(labels[class_samples])
                augmented_confs.append(confidences[class_samples])
            
            print(f"Class {class_id}: {len(class_samples)} → {len(class_samples) * (n_copies + 1)} samples")
    
    # Concatenate
    labels_aug = np.concatenate(augmented_labels, axis=0)
    confs_aug = np.concatenate(augmented_confs, axis=0)
    
    print(f"\nOriginal: {len(labels)} samples")
    print(f"Augmented: {len(labels_aug)} samples")
    
    # Save
    with open(output_path, 'wb') as f:
        pickle.dump({
            'labels': labels_aug,
            'confidences': confs_aug
        }, f)
    
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    augment_silver_labels(
        'data/intermediate/train_silver_labels.pkl',
        'data/intermediate/train_silver_labels_aug.pkl',
        target_min_samples=50
    )
