"""
Silver label generation using keyword matching and semantic similarity.
Based on TaxoClass approach with sentence-transformers.
"""

import numpy as np
import re
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import pickle
import os
import torch
from tqdm import tqdm


class SilverLabelGenerator:
    """Generate silver labels using keyword matching and semantic similarity."""
    
    def __init__(self, 
                 class_keywords: Dict[str, List[str]], 
                 class_to_id: Dict[str, int],
                 id_to_class: Dict[int, str],
                 min_confidence: float = 0.1,
                 embedding_model: str = "sentence-transformers/all-mpnet-base-v2",
                 use_keyword_matching: bool = True,
                 use_semantic_similarity: bool = True,
                 keyword_weight: float = 0.3,
                 similarity_weight: float = 0.7,
                 similarity_threshold: float = 0.5,
                 batch_size: int = 32,
                 device: Optional[str] = None):
        """
        Initialize silver label generator.
        
        Args:
            class_keywords: Dictionary mapping class names to keywords
            class_to_id: Class name to ID mapping
            id_to_class: ID to class name mapping
            min_confidence: Minimum confidence threshold for labeling
            embedding_model: Sentence transformer model name
            use_keyword_matching: Whether to use keyword matching
            use_semantic_similarity: Whether to use semantic similarity
            keyword_weight: Weight for keyword matching score
            similarity_weight: Weight for semantic similarity score
            similarity_threshold: Minimum cosine similarity threshold
            batch_size: Batch size for encoding
            device: Device to use (cuda/mps/cpu)
        """
        self.class_keywords = class_keywords
        self.class_to_id = class_to_id
        self.id_to_class = id_to_class
        self.num_classes = len(class_to_id)
        self.min_confidence = min_confidence
        
        self.use_keyword_matching = use_keyword_matching
        self.use_semantic_similarity = use_semantic_similarity
        self.keyword_weight = keyword_weight
        self.similarity_weight = similarity_weight
        self.similarity_threshold = similarity_threshold
        self.batch_size = batch_size
        
        # Setup device
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        # Preprocess keywords for matching
        if self.use_keyword_matching:
            self._preprocess_keywords()
        
        # Initialize semantic similarity model
        if self.use_semantic_similarity:
            self._initialize_embedding_model(embedding_model)
        
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
    
    def _initialize_embedding_model(self, model_name: str):
        """Initialize sentence transformer model for semantic similarity."""
        from sentence_transformers import SentenceTransformer
        
        print(f"Loading embedding model: {model_name}")
        self.embedding_model = SentenceTransformer(model_name, device=self.device)
        
        # Pre-encode class descriptions (class names + keywords)
        print("Encoding class descriptions...")
        class_texts = []
        self.class_ids_for_embeddings = []
        
        for class_name, keywords in self.class_keywords.items():
            # Create class description from name and keywords
            keywords_str = ", ".join(keywords[:10])  # Use first 10 keywords
            class_description = f"{class_name}: {keywords_str}"
            class_texts.append(class_description)
            self.class_ids_for_embeddings.append(self.class_to_id[class_name])
        
        # Encode all class descriptions
        self.class_embeddings = self.embedding_model.encode(
            class_texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            device=self.device
        )
        
        print(f"✓ Encoded {len(class_texts)} class descriptions")
    
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
    
    def compute_semantic_similarity(self, texts: List[str]) -> np.ndarray:
        """
        Compute semantic similarity between texts and class descriptions.
        
        Args:
            texts: List of input texts
            
        Returns:
            Similarity matrix (n_texts, n_classes)
        """
        from sentence_transformers import util
        
        # Encode texts
        text_embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_tensor=True,
            device=self.device
        )
        
        # Compute cosine similarity
        similarities = util.cos_sim(text_embeddings, self.class_embeddings)
        
        return similarities.cpu().numpy()
    
    def generate_labels_for_text(self, text: str, text_embedding: Optional[np.ndarray] = None) -> Dict[int, float]:
        """
        Generate labels for a single text using combined scoring.
        
        Args:
            text: Input text
            text_embedding: Pre-computed text embedding (if available)
            
        Returns:
            Dictionary mapping class IDs to confidence scores
        """
        combined_scores = {}
        
        # Keyword matching
        keyword_scores = {}
        if self.use_keyword_matching:
            keyword_scores = self.match_keywords(text)
        
        # Semantic similarity
        similarity_scores = {}
        if self.use_semantic_similarity and text_embedding is not None:
            for i, class_id in enumerate(self.class_ids_for_embeddings):
                sim_score = text_embedding[i]
                if sim_score >= self.similarity_threshold:
                    similarity_scores[class_id] = float(sim_score)
        
        # Combine scores
        all_class_ids = set(keyword_scores.keys()) | set(similarity_scores.keys())
        
        for class_id in all_class_ids:
            kw_score = keyword_scores.get(class_id, 0.0)
            sim_score = similarity_scores.get(class_id, 0.0)
            
            # Weighted combination
            if self.use_keyword_matching and self.use_semantic_similarity:
                combined_score = (self.keyword_weight * kw_score + 
                                self.similarity_weight * sim_score)
            elif self.use_keyword_matching:
                combined_score = kw_score
            else:
                combined_score = sim_score
            
            if combined_score >= self.min_confidence:
                combined_scores[class_id] = combined_score
        
        return combined_scores
    
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
        n_samples = len(corpus)
        labels = np.zeros((n_samples, self.num_classes), dtype=np.int32)
        confidences = np.zeros((n_samples, self.num_classes), dtype=np.float32)
        
        # Extract texts
        texts = [text for _, text in corpus]
        
        # Compute semantic similarities for all texts if enabled
        similarities = None
        if self.use_semantic_similarity:
            print("Computing semantic similarities...")
            similarities = self.compute_semantic_similarity(texts)
        
        # Generate labels
        labeled_count = 0
        keyword_only_count = 0
        semantic_only_count = 0
        both_count = 0
        
        for idx, (doc_id, text) in enumerate(tqdm(corpus, desc="Generating labels")):
            # Get text embedding if available
            text_embedding = similarities[idx] if similarities is not None else None
            
            # Generate combined scores
            class_scores = self.generate_labels_for_text(text, text_embedding)
            
            # Track labeling method
            has_keyword = False
            has_semantic = False
            
            if self.use_keyword_matching:
                kw_scores = self.match_keywords(text)
                has_keyword = len(kw_scores) > 0
            
            if self.use_semantic_similarity and text_embedding is not None:
                sem_scores = {self.class_ids_for_embeddings[i]: text_embedding[i] 
                             for i in range(len(self.class_ids_for_embeddings)) 
                             if text_embedding[i] >= self.similarity_threshold}
                has_semantic = len(sem_scores) > 0
            
            # Apply labels
            for class_id, score in class_scores.items():
                labels[idx, class_id] = 1
                confidences[idx, class_id] = score
            
            if np.sum(labels[idx]) > 0:
                labeled_count += 1
                if has_keyword and has_semantic:
                    both_count += 1
                elif has_keyword:
                    keyword_only_count += 1
                elif has_semantic:
                    semantic_only_count += 1
        
        coverage = labeled_count / n_samples * 100
        avg_labels = np.sum(labels) / labeled_count if labeled_count > 0 else 0
        
        print(f"\n=== Silver Labeling Results ===")
        print(f"Coverage: {coverage:.2f}% ({labeled_count}/{n_samples})")
        print(f"Average labels per document: {avg_labels:.2f}")
        print(f"Total labels assigned: {np.sum(labels)}")
        
        if self.use_keyword_matching and self.use_semantic_similarity:
            print(f"\nLabeling Method Breakdown:")
            print(f"  Keyword only: {keyword_only_count}")
            print(f"  Semantic only: {semantic_only_count}")
            print(f"  Both methods: {both_count}")
        
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