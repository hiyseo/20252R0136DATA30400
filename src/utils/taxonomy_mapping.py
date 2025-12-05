"""
Taxonomy mapping utilities for hierarchical label handling.
"""

import numpy as np
import networkx as nx
from typing import List, Set, Dict


class TaxonomyMapper:
    """Handles hierarchical label mapping and consistency."""
    
    def __init__(self, hierarchy: nx.DiGraph):
        self.hierarchy = hierarchy
        self.num_classes = hierarchy.number_of_nodes()
        
    def get_ancestors(self, class_id: int) -> Set[int]:
        """Get all ancestor classes (parents, grandparents, etc.)."""
        ancestors = set()
        for parent in self.hierarchy.predecessors(class_id):
            ancestors.add(parent)
            ancestors.update(self.get_ancestors(parent))
        return ancestors
    
    def get_descendants(self, class_id: int) -> Set[int]:
        """Get all descendant classes (children, grandchildren, etc.)."""
        descendants = set()
        for child in self.hierarchy.successors(class_id):
            descendants.add(child)
            descendants.update(self.get_descendants(child))
        return descendants
    
    def enforce_hierarchy(self, labels: np.ndarray) -> np.ndarray:
        """
        Ensure hierarchical consistency in labels.
        If a child class is labeled, all ancestors must be labeled too.
        
        Args:
            labels: Binary label vector (num_classes,)
            
        Returns:
            Hierarchically consistent label vector
        """
        consistent_labels = labels.copy()
        
        for class_id in np.where(labels == 1)[0]:
            # Add all ancestors
            ancestors = self.get_ancestors(class_id)
            for ancestor in ancestors:
                consistent_labels[ancestor] = 1
                
        return consistent_labels
    
    def labels_to_leaf_only(self, labels: np.ndarray) -> np.ndarray:
        """
        Convert hierarchical labels to leaf classes only.
        
        Args:
            labels: Binary label vector (num_classes,)
            
        Returns:
            Leaf-only label vector
        """
        leaf_labels = labels.copy()
        
        for class_id in np.where(labels == 1)[0]:
            # If this class has children that are also labeled, remove this class
            children = list(self.hierarchy.successors(class_id))
            if any(labels[child] == 1 for child in children):
                leaf_labels[class_id] = 0
                
        return leaf_labels
    
    def get_level(self, class_id: int) -> int:
        """Get the depth level of a class (root=0)."""
        if self.hierarchy.in_degree(class_id) == 0:
            return 0
        parents = list(self.hierarchy.predecessors(class_id))
        return 1 + max(self.get_level(p) for p in parents)
    
    def get_classes_at_level(self, level: int) -> List[int]:
        """Get all classes at a specific depth level."""
        return [node for node in self.hierarchy.nodes() 
                if self.get_level(node) == level]