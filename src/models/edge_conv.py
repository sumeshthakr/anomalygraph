"""
EdgeConv Layers for Dynamic Graph Neural Networks

Implements EdgeConv and Dynamic EdgeConv layers as described in:
"Dynamic Graph CNN for Learning on Point Clouds" (Wang et al., 2019)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, knn_graph
from torch_geometric.utils import add_self_loops
from typing import Optional, Tuple


class EdgeConvBlock(MessagePassing):
    """
    EdgeConv layer for point cloud feature learning.
    
    Computes edge features by concatenating node features with
    edge differences, then applies an MLP to transform them.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        hidden_channels: Hidden layer dimension (default: same as out_channels)
        aggr: Aggregation method ('max', 'mean', 'add')
        batch_norm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: Optional[int] = None,
        aggr: str = 'max',
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__(aggr=aggr)
        
        hidden_channels = hidden_channels or out_channels
        
        # MLP for edge feature transformation
        # Input: [x_i || x_j - x_i] -> 2 * in_channels
        self.mlp = nn.Sequential(
            nn.Linear(2 * in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of EdgeConv layer.
        
        Args:
            x: Node features (N, in_channels)
            edge_index: Graph connectivity (2, E)
            edge_attr: Optional edge features (not used in standard EdgeConv)
            
        Returns:
            Updated node features (N, out_channels)
        """
        return self.propagate(edge_index, x=x)
    
    def message(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        """
        Compute messages along edges.
        
        Args:
            x_i: Source node features (E, in_channels)
            x_j: Target node features (E, in_channels)
            
        Returns:
            Edge features (E, out_channels)
        """
        # Concatenate [x_i, x_j - x_i]
        edge_features = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(edge_features)


class EdgeConvWithEdgeFeatures(MessagePassing):
    """
    EdgeConv layer that incorporates explicit edge features.
    
    Useful when edge attributes (relative position, normal difference)
    are pre-computed in the graph.
    
    Args:
        in_channels: Input node feature dimension
        edge_channels: Input edge feature dimension
        out_channels: Output feature dimension
        aggr: Aggregation method
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int,
        edge_channels: int,
        out_channels: int,
        aggr: str = 'max',
        batch_norm: bool = True
    ):
        super().__init__(aggr=aggr)
        
        # MLP for combined node + edge features
        combined_dim = 2 * in_channels + edge_channels
        
        self.mlp = nn.Sequential(
            nn.Linear(combined_dim, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(
        self,
        x_i: torch.Tensor,
        x_j: torch.Tensor,
        edge_attr: torch.Tensor
    ) -> torch.Tensor:
        # Concatenate [x_i, x_j - x_i, edge_attr]
        edge_features = torch.cat([x_i, x_j - x_i, edge_attr], dim=-1)
        return self.mlp(edge_features)


class DynamicEdgeConv(nn.Module):
    """
    Dynamic EdgeConv that recomputes graph edges based on learned features.
    
    At each layer, kNN graph is recomputed in the feature space,
    allowing the network to learn dynamic neighborhood relationships.
    
    Args:
        in_channels: Input feature dimension
        out_channels: Output feature dimension
        k: Number of nearest neighbors
        aggr: Aggregation method
        batch_norm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        k: int = 20,
        aggr: str = 'max',
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.k = k
        self.edge_conv = EdgeConvBlock(
            in_channels, out_channels,
            aggr=aggr, batch_norm=batch_norm, dropout=dropout
        )
    
    def forward(
        self,
        x: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with dynamic graph construction.
        
        Args:
            x: Node features (N, in_channels)
            batch: Batch indices for each node
            pos: Optional positions for kNN (uses x if None)
            
        Returns:
            Tuple of (updated features, edge_index)
        """
        # Compute kNN graph in feature space
        knn_input = pos if pos is not None else x
        edge_index = knn_graph(knn_input, k=self.k, batch=batch, loop=False)
        
        # Apply EdgeConv
        x_out = self.edge_conv(x, edge_index)
        
        return x_out, edge_index


class EdgeConvEncoder(nn.Module):
    """
    Multi-layer EdgeConv encoder for point cloud feature extraction.
    
    Stacks multiple EdgeConv layers with residual connections
    and produces hierarchical features.
    
    Args:
        in_channels: Input feature dimension
        hidden_channels: List of hidden dimensions for each layer
        out_channels: Final output dimension
        k: Number of neighbors for kNN
        dynamic: Whether to use dynamic graph construction
        batch_norm: Whether to use batch normalization
        dropout: Dropout probability
    """
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: list = [64, 128, 256],
        out_channels: int = 256,
        k: int = 20,
        dynamic: bool = True,
        batch_norm: bool = True,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.dynamic = dynamic
        self.k = k
        
        # Build encoder layers
        channels = [in_channels] + hidden_channels
        self.layers = nn.ModuleList()
        
        for i in range(len(channels) - 1):
            if dynamic:
                layer = DynamicEdgeConv(
                    channels[i], channels[i + 1],
                    k=k, batch_norm=batch_norm, dropout=dropout
                )
            else:
                layer = EdgeConvBlock(
                    channels[i], channels[i + 1],
                    batch_norm=batch_norm, dropout=dropout
                )
            self.layers.append(layer)
        
        # Final projection
        total_channels = sum(hidden_channels)
        self.final_mlp = nn.Sequential(
            nn.Linear(total_channels, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.out_channels = out_channels
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        pos: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass through EdgeConv encoder.
        
        Args:
            x: Input features (N, in_channels)
            edge_index: Initial graph connectivity (optional if dynamic)
            batch: Batch indices
            pos: Point positions for dynamic kNN
            
        Returns:
            Node embeddings (N, out_channels)
        """
        features_list = []
        
        for layer in self.layers:
            if self.dynamic:
                x, edge_index = layer(x, batch=batch, pos=pos)
                # Update pos for next dynamic construction
                pos = x
            else:
                x = layer(x, edge_index)
            
            features_list.append(x)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(features_list, dim=-1)
        
        # Final projection
        out = self.final_mlp(multi_scale)
        
        return out
