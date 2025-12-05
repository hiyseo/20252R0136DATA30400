"""
Silver label generation using keyword matching.
Based on TaxoClass approach.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import pickle
import os


class SilverLabelGenerator:
    """Generate silver labels using keyword-based matching."""
    
    def __init__(self, class_keywords: Dict[str, List[str]], 
                 class_to_id: Dict[str, int],
                 id_to_class: Dict[int, str],
                 min_confidence: float = 0.1):
        """
        Initialize silver label generator.
        
        Args:
            class_keywords: Dictionary mapping class names to keywords
            class_to_id: Class name to ID mapping
            id_to_class: ID to class name mapping
            min_confidence: Minimum confidence threshold for labeling
        """
        self.class_keywords = class_keywords
        self.class_to_id = class_to_id
        self.id_to_class = id_to_class
        self.num_classes = len(class_to_id)
        self.min_confidence = min_confidence
        
        # Preprocess keywords for matching
        self._preprocess_keywords()
        
    def _preprocess_keywords(self):
        """Preprocess keywords for efficient matching."""
        self.keyword_patterns = {}
        self.compiled_patterns = {}
        
        for class_name, keywords in self.class_keywords.items():
            # Convert keywords to regex patterns and compile them
            patterns = []
            for kw in keywords:
                # Replace underscores with spaces or underscores
                kw_pattern = kw.replace('_', '[ _]')
                # Create word boundary pattern and compile
                pattern = re.compile(r'\b' + kw_pattern + r'\b', re.IGNORECASE)
                patterns.append(pattern)
            
            self.compiled_patterns[class_name] = patterns
    
    def match_keywords(self, text: str) -> Dict[int, float]:
        """
        Match keywords in text and return confidence scores.
        
        Args:
            text: Input text
            
        Returns:
            Dictionary mapping class IDs to confidence scores
        """
        class_scores = {}
        
        for class_name, patterns in self.compiled_patterns.items():
            matches = 0
            total_keywords = len(patterns)
            
            for pattern in patterns:
                if pattern.search(text):
                    matches += 1
            
            if matches > 0:
                # Confidence = proportion of matched keywords
                confidence = matches / total_keywords
                class_id = self.class_to_id[class_name]
                class_scores[class_id] = confidence
        
        return class_scores
    
    def generate_labels(self, corpus: List[Tuple[int, str]], 
                       output_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate silver labels for corpus.
        
        Args:
            corpus: List of (doc_id, text) tuples
            output_file: Optional file to save labels
            
        Returns:
            Tuple of (labels, confidences)
            - labels: Binary label matrix (n_samples, n_classes)
            - confidences: Confidence scores (n_samples, n_classes)
        """
        from tqdm import tqdm
        
        n_samples = len(corpus)
        labels = np.zeros((n_samples, self.num_classes), dtype=np.int32)
        confidences = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        
        labeled_count = 0
        
        for idx, (doc_id, text) in enumerate(tqdm(corpus, desc="Generating labels")):
            # Match keywords
            class_scores = self.match_keywords(text)
            
            # Apply confidence threshold
            for class_id, score in class_scores.items():
                if score >= self.min_confidence:
                    labels[idx, class_id] = 1
                    confidences[idx, class_id] = score
            
            if np.sum(labels[idx]) > 0:
                labeled_count += 1
        
        coverage = labeled_count / n_samples * 100
        avg_labels = np.sum(labels) / labeled_count if labeled_count > 0 else 0
        
        print(f"\n=== Silver Labeling Results ===")
        print(f"Coverage: {coverage:.2f}% ({labeled_count}/{n_samples})")
        print(f"Average labels per document: {avg_labels:.2f}")
        print(f"Total labels assigned: {np.sum(labels)}")
        
        # Save if output file specified
        if output_file:
            self._save_labels(labels, confidences, output_file)
        
        return labels, confidences
    
    def _save_labels(self, labels: np.ndarray, confidences: np.ndarray, 
                     output_file: str):
        """Save labels to file."""
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        data = {
            'labels': labels,
            'confidences': confidences
        }
        
        with open(output_file, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"✓ Labels saved to {output_file}")
    
    @staticmethod
    def load_labels(input_file: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load labels from file."""
        with open(input_file, 'rb') as f:
            data = pickle.load(f)
        
        return data['labels'], data['confidences']


if __name__ == "__main__":
    # Test silver label generation
    import sys
    sys.path.append('.')
    
    from src.data_preprocessing import DataLoader
    
    # Load data
    loader = DataLoader()
    loader.load_all()
    
    # Generate silver labels with lower threshold for better coverage
    generator = SilverLabelGenerator(
        loader.class_keywords,
        loader.class_to_id,
        loader.id_to_class,
        min_confidence=0.1  # Lower threshold for better coverage
    )
    
    print("\n=== Generating Silver Labels for Training Data ===")
    train_labels, train_confidences = generator.generate_labels(
        loader.train_corpus,
        output_file="data/intermediate/train_silver_labels.pkl"
    )
    
    print("\n=== Generating Silver Labels for Test Data ===")
    test_labels, test_confidences = generator.generate_labels(
        loader.test_corpus,
        output_file="data/intermediate/test_silver_labels.pkl"
    )