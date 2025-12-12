"""
Multi-label classifier with hierarchical constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class MultiLabelClassifier(nn.Module):
    """Simple multi-label classifier."""
    
    def __init__(self, input_size: int, num_classes: int, 
                 hidden_sizes: list = [512, 256],
                 dropout: float = 0.3):
        """
        Initialize classifier.
        
        Args:
            input_size: Input feature dimension
            num_classes: Number of output classes
            hidden_sizes: Hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, num_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_size)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        return self.network(x)


class HierarchicalClassifier(nn.Module):
    """Multi-label classifier with hierarchical awareness."""
    
    def __init__(self, input_size: int, num_classes: int,
                 hierarchy_matrix: Optional[np.ndarray] = None,
                 dropout: float = 0.3):
        """
        Initialize hierarchical classifier.
        
        Args:
            input_size: Input feature dimension
            num_classes: Number of output classes
            hierarchy_matrix: Binary matrix indicating parent-child relations
            dropout: Dropout rate
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Register hierarchy matrix as buffer (not a parameter)
        if hierarchy_matrix is not None:
            self.register_buffer(
                'hierarchy_matrix',
                torch.FloatTensor(hierarchy_matrix)
            )
        else:
            self.hierarchy_matrix = None
        
        # Feature extraction
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Class-specific classifiers
        self.classifier = nn.Linear(256, num_classes)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_size)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        features = self.feature_extractor(x)
        logits = self.classifier(features)
        
        return logits
    
    def predict_with_hierarchy(self, x: torch.Tensor, 
                               threshold: float = 0.5) -> torch.Tensor:
        """
        Predict with hierarchical constraint enforcement.
        
        Args:
            x: Input features (batch_size, input_size)
            threshold: Probability threshold
            
        Returns:
            Binary predictions (batch_size, num_classes)
        """
        logits = self.forward(x)
        probs = torch.sigmoid(logits)
        
        # Apply threshold
        preds = (probs >= threshold).float()
        
        # Enforce hierarchy: if child is 1, all ancestors must be 1
        if self.hierarchy_matrix is not None:
            # hierarchy_matrix[i, j] = 1 if i is ancestor of j
            ancestor_votes = torch.matmul(preds, self.hierarchy_matrix.T)
            # If any descendant is predicted, ancestor should be predicted
            preds = torch.clamp(preds + (ancestor_votes > 0).float(), max=1.0)
        
        return preds


def create_hierarchy_matrix(hierarchy, num_classes: int) -> np.ndarray:
    """
    Create ancestor matrix from hierarchy graph.
    
    Args:
        hierarchy: NetworkX DiGraph
        num_classes: Number of classes
        
    Returns:
        Binary matrix where [i,j]=1 if i is ancestor of j
    """
    import networkx as nx
    
    matrix = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    for child in range(num_classes):
        # Get all ancestors
        try:
            ancestors = nx.ancestors(hierarchy, child)
            for ancestor in ancestors:
                matrix[ancestor, child] = 1
        except nx.NetworkXError:
            continue
    
    return matrix


if __name__ == "__main__":
    # Test classifier
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MultiLabelClassifier(
        input_size=768,
        num_classes=531,
        hidden_sizes=[512, 256]
    ).to(device)
    
    # Test forward pass
    batch_size = 4
    x = torch.randn(batch_size, 768).to(device)
    
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✓ Classifier test passed!")