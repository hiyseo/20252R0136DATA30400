"""
Data preprocessing module for Amazon product classification.
Handles loading and preprocessing of corpus, classes, hierarchy, and keywords.
"""

import os
from typing import Dict, List, Tuple, Set
import networkx as nx
from collections import defaultdict


class DataLoader:
    """Load and preprocess Amazon product data."""
    
    def __init__(self, data_dir: str = "data/raw/Amazon_products"):
        self.data_dir = data_dir
        self.classes = []
        self.class_to_id = {}
        self.id_to_class = {}
        self.hierarchy = None
        self.class_keywords = {}
        self.train_corpus = []
        self.test_corpus = []
    
    @property
    def num_classes(self) -> int:
        """Return number of classes."""
        return len(self.class_to_id)
        
    def load_all(self):
        """Load all data files."""
        self.load_classes()
        self.load_hierarchy()
        self.load_keywords()
        self.load_train_corpus()
        self.load_test_corpus()
        
        print(f"✓ Loaded {len(self.classes)} classes")
        print(f"✓ Loaded hierarchy with {self.hierarchy.number_of_edges()} edges")
        print(f"✓ Loaded {len(self.train_corpus)} training samples")
        print(f"✓ Loaded {len(self.test_corpus)} test samples")
        
    def load_classes(self):
        """Load class names and create mappings."""
        filepath = os.path.join(self.data_dir, "classes.txt")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    class_id = int(parts[0])
                    class_name = parts[1]
                    self.classes.append(class_name)
                    self.class_to_id[class_name] = class_id
                    self.id_to_class[class_id] = class_name
                    
    def load_hierarchy(self):
        """Load class hierarchy as a directed graph."""
        filepath = os.path.join(self.data_dir, "class_hierarchy.txt")
        self.hierarchy = nx.DiGraph()
        
        # Add all nodes first
        for class_id in self.id_to_class.keys():
            self.hierarchy.add_node(class_id)
        
        # Add edges (parent -> child relationships)
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    parent_id = int(parts[0])
                    child_id = int(parts[1])
                    self.hierarchy.add_edge(parent_id, child_id)
                    
    def load_keywords(self):
        """Load class-related keywords."""
        filepath = os.path.join(self.data_dir, "class_related_keywords.txt")
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split(':')
                if len(parts) == 2:
                    class_name = parts[0]
                    keywords = [kw.strip() for kw in parts[1].split(',')]
                    self.class_keywords[class_name] = keywords
                    
    def load_train_corpus(self):
        """Load training corpus."""
        filepath = os.path.join(self.data_dir, "train/train_corpus.txt")
        self.train_corpus = self._load_corpus(filepath)
        
    def load_test_corpus(self):
        """Load test corpus."""
        filepath = os.path.join(self.data_dir, "test/test_corpus.txt")
        self.test_corpus = self._load_corpus(filepath)
        
    def _load_corpus(self, filepath: str) -> List[Tuple[int, str]]:
        """Load corpus file and return list of (doc_id, text) tuples."""
        corpus = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t', 1)
                if len(parts) == 2:
                    doc_id = int(parts[0])
                    text = parts[1]
                    corpus.append((doc_id, text))
        return corpus
    
    def get_parent_classes(self, class_id: int) -> Set[int]:
        """Get all parent classes for a given class."""
        parents = set()
        for parent in self.hierarchy.predecessors(class_id):
            parents.add(parent)
            parents.update(self.get_parent_classes(parent))
        return parents
    
    def get_child_classes(self, class_id: int) -> Set[int]:
        """Get all child classes for a given class."""
        children = set()
        for child in self.hierarchy.successors(class_id):
            children.add(child)
            children.update(self.get_child_classes(child))
        return children
    
    def get_class_level(self, class_id: int) -> int:
        """Get the depth level of a class in the hierarchy (root = 0)."""
        if self.hierarchy.in_degree(class_id) == 0:
            return 0
        parents = list(self.hierarchy.predecessors(class_id))
        return 1 + max(self.get_class_level(p) for p in parents)
    
    def get_root_classes(self) -> List[int]:
        """Get all root classes (classes with no parents)."""
        return [node for node in self.hierarchy.nodes() 
                if self.hierarchy.in_degree(node) == 0]
    
    def get_leaf_classes(self) -> List[int]:
        """Get all leaf classes (classes with no children)."""
        return [node for node in self.hierarchy.nodes() 
                if self.hierarchy.out_degree(node) == 0]


def preprocess_text(text: str) -> str:
    """Basic text preprocessing."""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text


if __name__ == "__main__":
    # Test data loading
    loader = DataLoader()
    loader.load_all()
    
    # Print some statistics
    print("\n=== Data Statistics ===")
    print(f"Number of classes: {len(loader.classes)}")
    print(f"Number of root classes: {len(loader.get_root_classes())}")
    print(f"Number of leaf classes: {len(loader.get_leaf_classes())}")
    
    # Sample class info
    sample_class_id = 0
    sample_class_name = loader.id_to_class[sample_class_id]
    print(f"\nSample class: {sample_class_name} (ID: {sample_class_id})")
    print(f"Keywords: {loader.class_keywords.get(sample_class_name, [])[:5]}")
    print(f"Children: {list(loader.get_child_classes(sample_class_id))[:5]}")
    
    # Sample documents
    print(f"\nSample training document:")
    doc_id, text = loader.train_corpus[0]
    print(f"Doc ID: {doc_id}")
    print(f"Text: {text[:200]}...")