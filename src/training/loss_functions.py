"""
Loss functions for hierarchical multi-label classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class BCEWithLogitsLoss(nn.Module):
    """Binary Cross Entropy loss for multi-label classification."""
    
    def __init__(self, pos_weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(logits, targets)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        """
        Args:
            alpha: Weighting factor
            gamma: Focusing parameter
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size, num_classes)
        """
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )
        
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        
        loss = alpha_weight * focal_weight * bce_loss
        
        return loss.mean()


class HierarchicalLoss(nn.Module):
    """Loss with hierarchical consistency constraint."""
    
    def __init__(self, hierarchy_matrix: torch.Tensor, 
                 lambda_hier: float = 0.1):
        """
        Args:
            hierarchy_matrix: (num_classes, num_classes) ancestor matrix
            lambda_hier: Weight for hierarchical constraint
        """
        super().__init__()
        self.register_buffer('hierarchy_matrix', hierarchy_matrix)
        self.lambda_hier = lambda_hier
        self.bce_loss = nn.BCEWithLogitsLoss()
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size, num_classes)
        """
        # Standard BCE loss
        bce = self.bce_loss(logits, targets)
        
        # Hierarchical constraint: if child is predicted, ancestor should be too
        probs = torch.sigmoid(logits)
        
        # For each class, compute max probability of its descendants
        # hierarchy_matrix[i, j] = 1 if i is ancestor of j
        descendant_probs = torch.matmul(probs, self.hierarchy_matrix.T)
        
        # Ancestors should have at least as high prob as descendants
        # Penalize: descendant_prob > ancestor_prob
        hier_violation = F.relu(descendant_probs - probs)
        hier_loss = hier_violation.mean()
        
        total_loss = bce + self.lambda_hier * hier_loss
        
        return total_loss


class AsymmetricLoss(nn.Module):
    """Asymmetric loss for multi-label classification."""
    
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 1, 
                 clip: float = 0.05):
        """
        Args:
            gamma_neg: Negative focusing parameter
            gamma_pos: Positive focusing parameter
            clip: Probability clipping value
        """
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (batch_size, num_classes)
            targets: (batch_size, num_classes)
        """
        # Probabilities
        probs = torch.sigmoid(logits)
        
        # Asymmetric clipping
        if self.clip is not None and self.clip > 0:
            probs = probs.clamp(min=self.clip)
        
        # Calculate losses
        targets = targets.float()
        
        # Positive samples
        pos_loss = targets * torch.log(probs)
        pos_loss = pos_loss * ((1 - probs) ** self.gamma_pos)
        
        # Negative samples
        neg_loss = (1 - targets) * torch.log(1 - probs)
        neg_loss = neg_loss * (probs ** self.gamma_neg)
        
        loss = -(pos_loss + neg_loss)
        
        return loss.mean()


def get_loss_function(loss_type: str, **kwargs) -> nn.Module:
    """
    Get loss function by name.
    
    Args:
        loss_type: 'bce', 'focal', 'hierarchical', or 'asymmetric'
        **kwargs: Additional arguments for loss function
        
    Returns:
        Loss function module
    """
    if loss_type == 'bce':
        return BCEWithLogitsLoss(**kwargs)
    elif loss_type == 'focal':
        return FocalLoss(**kwargs)
    elif loss_type == 'hierarchical':
        return HierarchicalLoss(**kwargs)
    elif loss_type == 'asymmetric':
        return AsymmetricLoss(**kwargs)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")