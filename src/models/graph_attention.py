"""
Graph Attention Layers with Edge Features

Implements Graph Attention Networks (GAT) enhanced with edge features
for geometry-aware point cloud processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import softmax
from typing import Optional, Tuple
import math


class GraphAttentionBlock(MessagePassing):
    """
    Graph Attention layer with edge feature support.
    
    Computes attention weights based on node features and edge attributes,
    allowing the network to focus on geometrically relevant neighbors.
    
    Args:
        in_channels: Input node feature dimension
        out_channels: Output feature dimension
        edge_channels: Edge feature dimension (0 if no edge features)
        heads: Number of attention heads
        concat: Whether to concatenate heads (True) or average (False)
        negative_slope: LeakyReLU negative slope
        dropout: Dropout probability
        bias: Whether to add bias
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        edge_channels: int = 0,
        heads: int = 4,
        concat: bool = True,
        negative_slope: float = 0.2,
        dropout: float = 0.0,
        bias: bool = True,
        batch_norm: bool = True
    ):
        super().__init__(aggr='add', node_dim=0)
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.edge_channels = edge_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        
        # Linear transformations for node features
        self.lin_src = nn.Linear(in_channels, heads * out_channels, bias=False)
        self.lin_dst = nn.Linear(in_channels, heads * out_channels, bias=False)
        
        # Attention parameters
        # att_src and att_dst for source and destination nodes
        self.att_src = nn.Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = nn.Parameter(torch.Tensor(1, heads, out_channels))
        
        # Edge feature transformation if provided
        if edge_channels > 0:
            self.lin_edge = nn.Linear(edge_channels, heads * out_channels, bias=False)
            self.att_edge = nn.Parameter(torch.Tensor(1, heads, out_channels))
        else:
            self.lin_edge = None
            self.att_edge = None
        
        if bias and concat:
            self.bias = nn.Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # Batch normalization
        if batch_norm:
            if concat:
                self.norm = nn.BatchNorm1d(heads * out_channels)
            else:
                self.norm = nn.BatchNorm1d(out_channels)
        else:
            self.norm = None
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_src.weight)
        nn.init.xavier_uniform_(self.lin_dst.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)
        
        if self.lin_edge is not None:
            nn.init.xavier_uniform_(self.lin_edge.weight)
            nn.init.xavier_uniform_(self.att_edge)
        
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of Graph Attention layer.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            edge_attr: Edge features (E, edge_channels)
            
        Returns:
            Updated node features (N, heads*out_channels or out_channels)
        """
        H, C = self.heads, self.out_channels
        
        # Transform node features
        x_src = self.lin_src(x).view(-1, H, C)
        x_dst = self.lin_dst(x).view(-1, H, C)
        
        # Compute attention scores for source and destination
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # (N, H)
        alpha_dst = (x_dst * self.att_dst).sum(dim=-1)  # (N, H)
        
        # Transform edge features if available
        if edge_attr is not None and self.lin_edge is not None:
            edge_feat = self.lin_edge(edge_attr).view(-1, H, C)
            alpha_edge = (edge_feat * self.att_edge).sum(dim=-1)  # (E, H)
        else:
            edge_feat = None
            alpha_edge = None
        
        # Propagate
        out = self.propagate(
            edge_index, 
            x=(x_src, x_dst),
            alpha=(alpha_src, alpha_dst),
            edge_feat=edge_feat,
            alpha_edge=alpha_edge
        )
        
        # Reshape output
        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)
        
        # Add bias
        if self.bias is not None:
            out = out + self.bias
        
        # Apply batch normalization
        if self.norm is not None:
            out = self.norm(out)
        
        return out
    
    def message(
        self,
        x_j: torch.Tensor,
        alpha_j: torch.Tensor,
        alpha_i: torch.Tensor,
        edge_feat: Optional[torch.Tensor],
        alpha_edge: Optional[torch.Tensor],
        index: torch.Tensor,
        ptr: Optional[torch.Tensor],
        size_i: Optional[int]
    ) -> torch.Tensor:
        """Compute attention-weighted messages."""
        # Compute attention coefficients
        alpha = alpha_j + alpha_i
        
        if alpha_edge is not None:
            alpha = alpha + alpha_edge
        
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        
        # Weighted message
        out = x_j * alpha.unsqueeze(-1)
        
        # Add edge features to message if available
        if edge_feat is not None:
            out = out + edge_feat * alpha.unsqueeze(-1)
        
        return out


class MultiHeadGraphAttention(nn.Module):
    """
    Multi-layer Graph Attention Network with residual connections.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: Hidden dimension
        out_channels: Output dimension
        num_layers: Number of attention layers
        heads: Number of attention heads per layer
        edge_channels: Edge feature dimension
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        heads: int = 4,
        edge_channels: int = 0,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(in_channels, hidden_channels)
        
        # Attention layers
        self.attention_layers = nn.ModuleList()
        self.skip_connections = nn.ModuleList()
        
        for i in range(num_layers):
            # Each layer outputs heads * hidden_channels // heads = hidden_channels
            layer = GraphAttentionBlock(
                in_channels=hidden_channels,
                out_channels=hidden_channels // heads,
                edge_channels=edge_channels,
                heads=heads,
                concat=True,
                dropout=dropout,
                batch_norm=batch_norm
            )
            self.attention_layers.append(layer)
            
            # Skip connection
            self.skip_connections.append(nn.Linear(hidden_channels, hidden_channels))
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through multi-layer GAT.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity
            edge_attr: Edge features
            
        Returns:
            Node embeddings (N, out_channels)
        """
        # Input projection
        x = self.input_proj(x)
        
        # Apply attention layers with residual connections
        for i, (attn, skip) in enumerate(zip(self.attention_layers, self.skip_connections)):
            identity = x
            x = attn(x, edge_index, edge_attr)
            x = self.dropout(x)
            x = x + skip(identity)  # Residual connection
            x = F.relu(x)
        
        # Output projection
        x = self.output_proj(x)
        
        return x


class GeometryAwareAttention(nn.Module):
    """
    Geometry-aware attention that explicitly models spatial relationships.
    
    Combines position-based attention with feature-based attention
    for better geometric reasoning.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output dimension
        pos_channels: Position dimension (usually 3 for xyz)
        heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        pos_channels: int = 3,
        heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads
        
        # Feature transformations
        self.query = nn.Linear(in_channels, out_channels)
        self.key = nn.Linear(in_channels, out_channels)
        self.value = nn.Linear(in_channels, out_channels)
        
        # Position encoding
        self.pos_encoder = nn.Sequential(
            nn.Linear(pos_channels, out_channels // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 2, out_channels)
        )
        
        # Relative position attention
        self.rel_pos_key = nn.Linear(pos_channels, out_channels)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.head_dim)
        
        self.output_proj = nn.Linear(out_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        pos: torch.Tensor,
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass with geometry-aware attention.
        
        Args:
            x: Node features (N, in_channels)
            pos: Node positions (N, pos_channels)
            edge_index: Graph connectivity (2, E)
            
        Returns:
            Updated node features (N, out_channels)
        """
        src, dst = edge_index
        N = x.size(0)
        H = self.heads
        D = self.head_dim
        
        # Compute Q, K, V
        Q = self.query(x).view(N, H, D)
        K = self.key(x).view(N, H, D)
        V = self.value(x).view(N, H, D)
        
        # Relative positions
        rel_pos = pos[dst] - pos[src]  # (E, 3)
        rel_pos_encoded = self.rel_pos_key(rel_pos).view(-1, H, D)
        
        # Compute attention scores
        Q_src = Q[src]  # (E, H, D)
        K_dst = K[dst]  # (E, H, D)
        
        # Attention = Q * (K + rel_pos_encoding) / sqrt(d)
        attn_scores = (Q_src * (K_dst + rel_pos_encoded)).sum(dim=-1) / self.scale
        
        # Softmax over neighbors
        attn_weights = softmax(attn_scores, src, num_nodes=N)
        attn_weights = self.dropout(attn_weights)
        
        # Weighted aggregation
        V_dst = V[dst]  # (E, H, D)
        messages = V_dst * attn_weights.unsqueeze(-1)
        
        # Aggregate
        out = torch.zeros(N, H, D, device=x.device)
        out.scatter_add_(0, src.view(-1, 1, 1).expand(-1, H, D), messages)
        
        # Reshape and project
        out = out.view(N, -1)
        out = self.output_proj(out)
        
        return out
