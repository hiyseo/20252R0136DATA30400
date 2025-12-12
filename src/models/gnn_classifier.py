"""
Graph Neural Network classifier for hierarchical classification.
Uses taxonomy structure to propagate information between classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import networkx as nx


class GraphConvolution(nn.Module):
    """Simple Graph Convolutional Layer."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_features)
            adj: (num_nodes, num_nodes) adjacency matrix
        Returns:
            (num_nodes, out_features)
        """
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support) if adj.is_sparse else torch.mm(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        return output


class GCNClassifier(nn.Module):
    """GCN-based hierarchical classifier."""
    
    def __init__(self,
                 text_encoder: nn.Module,
                 num_classes: int,
                 hidden_dim: int = 512,
                 num_gcn_layers: int = 2,
                 dropout: float = 0.3,
                 adjacency_matrix: Optional[torch.Tensor] = None):
        """
        Initialize GCN classifier.
        
        Args:
            text_encoder: Pre-trained text encoder (e.g., BERT)
            num_classes: Number of classes
            hidden_dim: Hidden dimension for GCN
            num_gcn_layers: Number of GCN layers
            dropout: Dropout rate
            adjacency_matrix: Adjacency matrix of taxonomy (num_classes, num_classes)
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        
        # Text encoding dimension (e.g., 768 for BERT)
        self.text_dim = text_encoder.hidden_size
        
        # Project text features to class space
        self.text_to_class = nn.Linear(self.text_dim, num_classes)
        
        # Class embeddings (learnable)
        self.class_embeddings = nn.Parameter(torch.FloatTensor(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.class_embeddings)
        
        # GCN layers
        self.gcn_layers = nn.ModuleList()
        for i in range(num_gcn_layers):
            in_dim = hidden_dim if i == 0 else hidden_dim
            self.gcn_layers.append(GraphConvolution(in_dim, hidden_dim))
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        
        self.dropout = nn.Dropout(dropout)
        
        # Register adjacency matrix
        if adjacency_matrix is not None:
            self.register_buffer('adj', adjacency_matrix)
        else:
            # Identity matrix if no adjacency provided
            self.register_buffer('adj', torch.eye(num_classes))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            input_ids: (batch_size, seq_len)
            attention_mask: (batch_size, seq_len)
            
        Returns:
            logits: (batch_size, num_classes)
        """
        batch_size = input_ids.size(0)
        
        # Encode text
        text_features = self.text_encoder(input_ids, attention_mask)  # (batch, text_dim)
        
        # Initial class predictions from text
        text_logits = self.text_to_class(text_features)  # (batch, num_classes)
        
        # GCN on class embeddings
        class_features = self.class_embeddings  # (num_classes, hidden_dim)
        
        for gcn_layer in self.gcn_layers:
            class_features = gcn_layer(class_features, self.adj)
            class_features = F.relu(class_features)
            class_features = self.dropout(class_features)
        
        # Project class features to scores
        class_scores = self.output_proj(class_features).squeeze(-1)  # (num_classes,)
        
        # Combine text predictions with GCN class scores
        # Text logits provide instance-specific information
        # Class scores provide taxonomy structure information
        logits = text_logits + class_scores.unsqueeze(0)  # (batch, num_classes)
        
        return logits


class GATLayer(nn.Module):
    """Graph Attention Layer."""
    
    def __init__(self, in_features: int, out_features: int, dropout: float = 0.3, alpha: float = 0.2):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))
        
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (num_nodes, in_features)
            adj: (num_nodes, num_nodes) adjacency matrix
        Returns:
            (num_nodes, out_features)
        """
        # Linear transformation
        h = torch.mm(x, self.W)  # (N, out_features)
        N = h.size(0)
        
        # Attention mechanism
        a_input = torch.cat([h.repeat(1, N).view(N * N, -1),
                            h.repeat(N, 1)], dim=1).view(N, N, 2 * self.out_features)
        
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Mask attention with adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=1)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention
        h_prime = torch.matmul(attention, h)
        
        return h_prime


class GATClassifier(nn.Module):
    """GAT-based hierarchical classifier."""
    
    def __init__(self,
                 text_encoder: nn.Module,
                 num_classes: int,
                 hidden_dim: int = 512,
                 num_gat_layers: int = 2,
                 num_heads: int = 4,
                 dropout: float = 0.3,
                 adjacency_matrix: Optional[torch.Tensor] = None):
        """
        Initialize GAT classifier.
        
        Args:
            text_encoder: Pre-trained text encoder
            num_classes: Number of classes
            hidden_dim: Hidden dimension
            num_gat_layers: Number of GAT layers
            num_heads: Number of attention heads
            dropout: Dropout rate
            adjacency_matrix: Adjacency matrix of taxonomy
        """
        super().__init__()
        
        self.text_encoder = text_encoder
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        self.text_dim = text_encoder.hidden_size
        
        # Text to class projection
        self.text_to_class = nn.Linear(self.text_dim, num_classes)
        
        # Class embeddings
        self.class_embeddings = nn.Parameter(torch.FloatTensor(num_classes, hidden_dim))
        nn.init.xavier_uniform_(self.class_embeddings)
        
        # Multi-head GAT layers
        self.gat_layers = nn.ModuleList()
        for i in range(num_gat_layers):
            layer_heads = nn.ModuleList()
            for _ in range(num_heads):
                layer_heads.append(GATLayer(hidden_dim, hidden_dim // num_heads, dropout))
            self.gat_layers.append(layer_heads)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Register adjacency
        if adjacency_matrix is not None:
            self.register_buffer('adj', adjacency_matrix)
        else:
            self.register_buffer('adj', torch.eye(num_classes))
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Encode text
        text_features = self.text_encoder(input_ids, attention_mask)
        text_logits = self.text_to_class(text_features)
        
        # GAT on class embeddings
        class_features = self.class_embeddings
        
        for gat_heads in self.gat_layers:
            # Multi-head attention
            head_outputs = [head(class_features, self.adj) for head in gat_heads]
            class_features = torch.cat(head_outputs, dim=-1)
            class_features = F.elu(class_features)
            class_features = self.dropout(class_features)
        
        # Project to scores
        class_scores = self.output_proj(class_features).squeeze(-1)
        
        # Combine
        logits = text_logits + class_scores.unsqueeze(0)
        
        return logits


def build_adjacency_matrix(hierarchy_graph: nx.DiGraph, num_classes: int, self_loop: bool = True) -> torch.Tensor:
    """
    Build adjacency matrix from hierarchy graph.
    Use graph_utils for numpy version, then convert to torch.
    
    Args:
        hierarchy_graph: NetworkX DiGraph of taxonomy
        num_classes: Number of classes
        self_loop: Whether to add self-loops
        
    Returns:
        Normalized adjacency matrix (num_classes, num_classes)
    """
    # Import from graph_utils to avoid duplication
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from silver_labeling.graph_utils import build_adjacency_matrix_from_graph
    
    # Use graph_utils implementation
    adj_np = build_adjacency_matrix_from_graph(
        hierarchy_graph, 
        num_classes, 
        normalize=True, 
        add_self_loops=self_loop
    )
    
    return torch.FloatTensor(adj_np)