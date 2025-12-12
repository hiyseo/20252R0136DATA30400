"""
Taxonomy graph utilities for hierarchical classification.
Provides functions for analyzing and processing class hierarchies.
"""

import numpy as np
import networkx as nx
from typing import List, Set, Dict, Tuple, Optional
from collections import defaultdict, deque


class TaxonomyGraph:
    """Wrapper for taxonomy graph operations."""
    
    def __init__(self, hierarchy_graph: nx.DiGraph, num_classes: int):
        """
        Initialize taxonomy graph.
        
        Args:
            hierarchy_graph: NetworkX directed graph representing taxonomy
            num_classes: Total number of classes
        """
        self.graph = hierarchy_graph
        self.num_classes = num_classes
        
        # Cache for expensive computations
        self._ancestor_cache = {}
        self._descendant_cache = {}
        self._depth_cache = {}
        self._ancestor_matrix = None
    
    def get_ancestors(self, node_id: int, include_self: bool = False) -> Set[int]:
        """
        Get all ancestors of a node (parents, grandparents, etc.).
        
        Args:
            node_id: Node ID
            include_self: Whether to include the node itself
            
        Returns:
            Set of ancestor node IDs
        """
        if node_id in self._ancestor_cache:
            ancestors = self._ancestor_cache[node_id].copy()
        else:
            ancestors = set()
            if node_id in self.graph:
                ancestors = nx.ancestors(self.graph, node_id)
            self._ancestor_cache[node_id] = ancestors.copy()
        
        if include_self:
            ancestors.add(node_id)
        
        return ancestors
    
    def get_descendants(self, node_id: int, include_self: bool = False) -> Set[int]:
        """
        Get all descendants of a node (children, grandchildren, etc.).
        
        Args:
            node_id: Node ID
            include_self: Whether to include the node itself
            
        Returns:
            Set of descendant node IDs
        """
        if node_id in self._descendant_cache:
            descendants = self._descendant_cache[node_id].copy()
        else:
            descendants = set()
            if node_id in self.graph:
                descendants = nx.descendants(self.graph, node_id)
            self._descendant_cache[node_id] = descendants.copy()
        
        if include_self:
            descendants.add(node_id)
        
        return descendants
    
    def get_parents(self, node_id: int) -> List[int]:
        """Get immediate parents of a node."""
        if node_id not in self.graph:
            return []
        return list(self.graph.predecessors(node_id))
    
    def get_children(self, node_id: int) -> List[int]:
        """Get immediate children of a node."""
        if node_id not in self.graph:
            return []
        return list(self.graph.successors(node_id))
    
    def get_siblings(self, node_id: int) -> Set[int]:
        """Get sibling nodes (nodes with same parent)."""
        siblings = set()
        parents = self.get_parents(node_id)
        for parent in parents:
            siblings.update(self.get_children(parent))
        siblings.discard(node_id)  # Remove self
        return siblings
    
    def get_depth(self, node_id: int) -> int:
        """
        Get depth of a node (distance from root).
        
        Args:
            node_id: Node ID
            
        Returns:
            Depth (0 for root nodes)
        """
        if node_id in self._depth_cache:
            return self._depth_cache[node_id]
        
        if node_id not in self.graph:
            return 0
        
        # Find roots (nodes with no predecessors)
        roots = [n for n in self.graph.nodes() if self.graph.in_degree(n) == 0]
        
        if node_id in roots:
            depth = 0
        else:
            # BFS from roots to find shortest path
            depth = float('inf')
            for root in roots:
                try:
                    path_length = nx.shortest_path_length(self.graph, root, node_id)
                    depth = min(depth, path_length)
                except nx.NetworkXNoPath:
                    continue
            
            if depth == float('inf'):
                depth = 0
        
        self._depth_cache[node_id] = depth
        return depth
    
    def is_leaf(self, node_id: int) -> bool:
        """Check if node is a leaf (no children)."""
        return node_id in self.graph and self.graph.out_degree(node_id) == 0
    
    def is_root(self, node_id: int) -> bool:
        """Check if node is a root (no parents)."""
        return node_id in self.graph and self.graph.in_degree(node_id) == 0
    
    def get_common_ancestor(self, node1: int, node2: int) -> Optional[int]:
        """
        Find lowest common ancestor of two nodes.
        
        Args:
            node1: First node ID
            node2: Second node ID
            
        Returns:
            Lowest common ancestor ID or None
        """
        ancestors1 = self.get_ancestors(node1, include_self=True)
        ancestors2 = self.get_ancestors(node2, include_self=True)
        
        common = ancestors1 & ancestors2
        if not common:
            return None
        
        # Find the one with maximum depth (lowest in tree)
        lca = max(common, key=lambda n: self.get_depth(n))
        return lca
    
    def get_path(self, source: int, target: int) -> Optional[List[int]]:
        """
        Get shortest path between two nodes.
        
        Args:
            source: Source node ID
            target: Target node ID
            
        Returns:
            List of node IDs in path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None
    
    def build_ancestor_matrix(self) -> np.ndarray:
        """
        Build ancestor matrix for hierarchical loss.
        Matrix[i, j] = 1 if i is ancestor of j, 0 otherwise.
        
        Returns:
            Ancestor matrix (num_classes, num_classes)
        """
        if self._ancestor_matrix is not None:
            return self._ancestor_matrix
        
        matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.float32)
        
        for node_id in range(self.num_classes):
            ancestors = self.get_ancestors(node_id, include_self=False)
            for ancestor in ancestors:
                if ancestor < self.num_classes:
                    matrix[ancestor, node_id] = 1.0
        
        self._ancestor_matrix = matrix
        return matrix
    
    def get_hierarchy_stats(self) -> Dict:
        """
        Compute statistics about the hierarchy.
        
        Returns:
            Dictionary with statistics
        """
        roots = [n for n in self.graph.nodes() if self.is_root(n)]
        leaves = [n for n in self.graph.nodes() if self.is_leaf(n)]
        
        depths = [self.get_depth(n) for n in self.graph.nodes()]
        
        stats = {
            'num_nodes': self.graph.number_of_nodes(),
            'num_edges': self.graph.number_of_edges(),
            'num_roots': len(roots),
            'num_leaves': len(leaves),
            'max_depth': max(depths) if depths else 0,
            'avg_depth': np.mean(depths) if depths else 0,
            'avg_children': np.mean([self.graph.out_degree(n) for n in self.graph.nodes()]),
        }
        
        return stats
    
    def expand_labels_with_hierarchy(self, labels: np.ndarray) -> np.ndarray:
        """
        Expand labels to include ancestors (positive propagation).
        If a class is labeled, all its ancestors should also be labeled.
        
        Args:
            labels: Binary label matrix (n_samples, num_classes)
            
        Returns:
            Expanded label matrix
        """
        expanded = labels.copy()
        
        for sample_idx in range(labels.shape[0]):
            positive_classes = np.where(labels[sample_idx] == 1)[0]
            
            # Add all ancestors
            for class_id in positive_classes:
                ancestors = self.get_ancestors(class_id, include_self=False)
                for ancestor in ancestors:
                    if ancestor < self.num_classes:
                        expanded[sample_idx, ancestor] = 1
        
        return expanded
    
    def filter_negative_children(self, labels: np.ndarray) -> np.ndarray:
        """
        Remove children of positive classes from negative set.
        If a parent is labeled, don't penalize predictions of its children.
        
        Args:
            labels: Binary label matrix (n_samples, num_classes)
            
        Returns:
            Filtered label matrix (sets children to -1 to ignore in loss)
        """
        filtered = labels.copy()
        
        for sample_idx in range(labels.shape[0]):
            positive_classes = np.where(labels[sample_idx] == 1)[0]
            
            # Find all descendants
            descendants_to_ignore = set()
            for class_id in positive_classes:
                descendants = self.get_descendants(class_id, include_self=False)
                descendants_to_ignore.update(descendants)
            
            # Set descendants to -1 (ignore)
            for desc in descendants_to_ignore:
                if desc < self.num_classes and labels[sample_idx, desc] == 0:
                    filtered[sample_idx, desc] = -1
        
        return filtered


def build_adjacency_matrix_from_graph(graph: nx.DiGraph, 
                                      num_classes: int, 
                                      normalize: bool = True,
                                      add_self_loops: bool = True) -> np.ndarray:
    """
    Build adjacency matrix from NetworkX graph for GNN.
    
    Args:
        graph: NetworkX directed graph
        num_classes: Number of classes
        normalize: Whether to apply symmetric normalization
        add_self_loops: Whether to add self-loops
        
    Returns:
        Adjacency matrix (num_classes, num_classes)
    """
    # Initialize adjacency matrix
    adj = np.zeros((num_classes, num_classes), dtype=np.float32)
    
    # Add edges (make undirected by adding both directions)
    for parent, child in graph.edges():
        if parent < num_classes and child < num_classes:
            adj[parent, child] = 1.0
            adj[child, parent] = 1.0  # Undirected
    
    # Add self-loops
    if add_self_loops:
        adj = adj + np.eye(num_classes, dtype=np.float32)
    
    # Normalize: D^(-1/2) * A * D^(-1/2)
    if normalize:
        degree = adj.sum(axis=1)
        degree_inv_sqrt = np.power(degree, -0.5)
        degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.0
        
        D_inv_sqrt = np.diag(degree_inv_sqrt)
        adj = D_inv_sqrt @ adj @ D_inv_sqrt
    
    return adj


def visualize_subgraph(graph: nx.DiGraph, 
                       node_ids: List[int],
                       class_names: Optional[Dict[int, str]] = None) -> nx.DiGraph:
    """
    Extract subgraph for visualization.
    
    Args:
        graph: Full taxonomy graph
        node_ids: Node IDs to include
        class_names: Optional mapping from ID to class name
        
    Returns:
        Subgraph
    """
    subgraph = graph.subgraph(node_ids).copy()
    
    # Add class names as labels if provided
    if class_names:
        nx.set_node_attributes(subgraph, 
                              {n: class_names.get(n, str(n)) for n in subgraph.nodes()},
                              'label')
    
    return subgraph