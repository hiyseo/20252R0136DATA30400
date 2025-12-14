"""
PyTorch Dataset for hierarchical multi-label classification.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import numpy as np
import pickle
from typing import List, Tuple, Optional


class ProductReviewDataset(Dataset):
    """Dataset for product review classification."""
    
    def __init__(self, 
                 corpus: List[Tuple[int, str]],
                 labels: Optional[np.ndarray] = None,
                 tokenizer_name: str = "bert-base-uncased",  # Keep default for backward compatibility
                 max_length: int = 128,
                 use_cached_tokenizer: bool = True):
        """
        Initialize dataset.
        
        Args:
            corpus: List of (doc_id, text) tuples
            labels: Optional label matrix (n_samples, n_classes)
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            use_cached_tokenizer: Whether to use cached tokenizer
        """
        self.corpus = corpus
        self.labels = labels
        self.max_length = max_length
        
        if use_cached_tokenizer:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name, 
                use_fast=True
            )
    
    def __len__(self) -> int:
        return len(self.corpus)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single item.
        
        Returns:
            Dictionary with keys: input_ids, attention_mask, labels (optional), doc_id
        """
        doc_id, text = self.corpus[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'doc_id': doc_id
        }
        
        # Add labels if available
        if self.labels is not None:
            item['labels'] = torch.FloatTensor(self.labels[idx])
        
        return item


class SilverLabelDataset(ProductReviewDataset):
    """Dataset with silver labels and confidence scores."""
    
    def __init__(self,
                 corpus: List[Tuple[int, str]],
                 labels: np.ndarray,
                 confidences: np.ndarray,
                 tokenizer_name: str = "bert-base-uncased",  # Keep default for backward compatibility
                 max_length: int = 128,
                 min_confidence: float = 0.0):
        """
        Initialize dataset with confidence filtering.
        
        Args:
            corpus: List of (doc_id, text) tuples
            labels: Label matrix (n_samples, n_classes)
            confidences: Confidence matrix (n_samples, n_classes)
            tokenizer_name: HuggingFace tokenizer name
            max_length: Maximum sequence length
            min_confidence: Minimum confidence threshold
        """
        super().__init__(corpus, labels, tokenizer_name, max_length)
        
        self.confidences = confidences
        self.min_confidence = min_confidence
        
        # Filter samples with at least one confident label
        self.valid_indices = self._get_valid_indices()
        
    def _get_valid_indices(self) -> List[int]:
        """Get indices of samples with at least one confident label."""
        valid = []
        for idx in range(len(self.corpus)):
            max_conf = np.max(self.confidences[idx])
            if max_conf >= self.min_confidence:
                valid.append(idx)
        return valid
    
    def __len__(self) -> int:
        return len(self.valid_indices)
    
    def __getitem__(self, idx: int) -> dict:
        """Get item with confidence scores."""
        # Map to original index
        original_idx = self.valid_indices[idx]
        
        item = super().__getitem__(original_idx)
        item['confidences'] = torch.FloatTensor(self.confidences[original_idx])
        
        return item


def create_dataloaders(train_corpus: List[Tuple[int, str]],
                      test_corpus: List[Tuple[int, str]],
                      train_labels: np.ndarray,
                      train_confidences: Optional[np.ndarray] = None,
                      test_labels: Optional[np.ndarray] = None,
                      test_confidences: Optional[np.ndarray] = None,
                      tokenizer_name: str = "bert-base-uncased",  # Keep default for backward compatibility
                      batch_size: int = 16,
                      max_length: int = 128,
                      num_workers: int = 4):
    """
    Create train and test dataloaders.
    
    Args:
        train_corpus: Training corpus
        test_corpus: Test corpus
        train_labels: Training labels
        train_confidences: Optional training confidence scores
        test_labels: Optional test silver labels (for self-training)
        test_confidences: Optional test confidence scores
        tokenizer_name: Tokenizer name
        batch_size: Batch size
        max_length: Max sequence length
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    from torch.utils.data import DataLoader
    
    # Create datasets
    if train_confidences is not None:
        train_dataset = SilverLabelDataset(
            train_corpus,
            train_labels,
            train_confidences,
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )
    else:
        train_dataset = ProductReviewDataset(
            train_corpus,
            train_labels,
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )
    
    # Create test dataset with silver labels if available
    if test_labels is not None and test_confidences is not None:
        test_dataset = SilverLabelDataset(
            test_corpus,
            test_labels,
            test_confidences,
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )
        print("✓ Test dataset created with silver labels")
    else:
        test_dataset = ProductReviewDataset(
            test_corpus,
            labels=None,
            tokenizer_name=tokenizer_name,
            max_length=max_length
        )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset
    import sys
    sys.path.append('.')
    
    from src.data_preprocessing import DataLoader
    from src.silver_labeling.generate_silver_labels import SilverLabelGenerator
    
    # Load data
    loader = DataLoader()
    loader.load_all()
    
    # Load silver labels
    train_labels, train_confidences = SilverLabelGenerator.load_labels(
        "data/intermediate/train_silver_labels.pkl"
    )
    
    # Create dataset
    dataset = SilverLabelDataset(
        loader.train_corpus[:100],  # Test with 100 samples
        train_labels[:100],
        train_confidences[:100],
        tokenizer_name="bert-base-uncased",
        max_length=128
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Test a single item
    item = dataset[0]
    print(f"\nSample item:")
    print(f"  input_ids shape: {item['input_ids'].shape}")
    print(f"  attention_mask shape: {item['attention_mask'].shape}")
    print(f"  labels shape: {item['labels'].shape}")
    print(f"  confidences shape: {item['confidences'].shape}")
    print(f"  doc_id: {item['doc_id']}")
    print(f"  num_labels: {item['labels'].sum().item()}")
    
    print("\n✓ Dataset test passed!")
