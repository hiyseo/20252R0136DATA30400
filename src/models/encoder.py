"""
Text encoder using pretrained language models (BERT, RoBERTa, etc.).
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from typing import Dict, List


class TextEncoder(nn.Module):
    """Pretrained language model encoder for text classification."""
    
    def __init__(self, model_name: str, 
                 num_classes: int = 531,
                 dropout: float = 0.1,
                 freeze_encoder: bool = False):
        """
        Initialize text encoder.
        
        Args:
            model_name: HuggingFace model name
            num_classes: Number of output classes
            dropout: Dropout rate
            freeze_encoder: Whether to freeze pretrained weights
        """
        super().__init__()
        
        self.encoder = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.encoder.config.hidden_size
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.hidden_size, num_classes)
        
    def forward(self, input_ids: torch.Tensor, 
                attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Logits (batch_size, num_classes)
        """
        # Get encoder output
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]
        
        # Classification
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits
    
    def get_embeddings(self, input_ids: torch.Tensor,
                      attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Get text embeddings without classification.
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            
        Returns:
            Embeddings (batch_size, hidden_size)
        """
        with torch.no_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            embeddings = outputs.last_hidden_state[:, 0, :]
        
        return embeddings


def get_device() -> torch.device:
    """Get available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def load_tokenizer(model_name: str = "bert-base-uncased"):
    """Load tokenizer for the model."""
    return AutoTokenizer.from_pretrained(model_name)


if __name__ == "__main__":
    # Test encoder
    device = get_device()
    print(f"Using device: {device}")
    
    model = TextEncoder(model_name="bert-base-uncased", num_classes=531)
    model = model.to(device)
    
    # Test forward pass
    tokenizer = load_tokenizer("bert-base-uncased")
    texts = ["This is a test sentence.", "Another example text."]
    
    encoded = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)
    
    with torch.no_grad():
        logits = model(input_ids, attention_mask)
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"✓ Encoder test passed!")