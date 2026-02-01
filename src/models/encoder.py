"""
Hierarchical Graph Encoder for 3D Point Cloud Feature Extraction

Implements a two-level hierarchical encoder:
1. Fine-level: Processes all points for micro-defect detection
2. Coarse-level: Processes superpoints for macro-defect detection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import global_mean_pool, global_max_pool
from typing import Optional, Tuple, Dict

from .edge_conv import EdgeConvBlock, EdgeConvEncoder
from .graph_attention import MultiHeadGraphAttention, GeometryAwareAttention


class HierarchicalGraphEncoder(nn.Module):
    """
    Hierarchical Graph Neural Network Encoder for point clouds.
    
    Architecture:
    1. Fine-level encoder: Processes all points
    2. Pooling: Aggregates features to superpoints
    3. Coarse-level encoder: Processes superpoint graph
    4. Global pooling: Produces global embedding
    
    Args:
        in_channels: Input feature dimension (8 for [xyz, normals, curvature, roughness])
        hidden_channels: Hidden dimension for fine-level
        coarse_hidden: Hidden dimension for coarse-level
        out_channels: Output embedding dimension
        fine_layers: Number of fine-level layers
        coarse_layers: Number of coarse-level layers
        encoder_type: 'edgeconv' or 'attention'
        k_fine: Number of neighbors for fine graph
        k_coarse: Number of neighbors for coarse graph
        heads: Attention heads (if using attention)
        dropout: Dropout probability
        batch_norm: Whether to use batch normalization
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 128,
        coarse_hidden: int = 256,
        out_channels: int = 256,
        fine_layers: int = 3,
        coarse_layers: int = 2,
        encoder_type: str = 'edgeconv',
        k_fine: int = 24,
        k_coarse: int = 16,
        heads: int = 4,
        dropout: float = 0.1,
        batch_norm: bool = True
    ):
        super().__init__()
        
        self.encoder_type = encoder_type
        self.out_channels = out_channels
        
        # Fine-level encoder
        if encoder_type == 'edgeconv':
            fine_hidden_list = [hidden_channels] * fine_layers
            self.fine_encoder = EdgeConvEncoder(
                in_channels=in_channels,
                hidden_channels=fine_hidden_list,
                out_channels=hidden_channels,
                k=k_fine,
                dynamic=True,
                batch_norm=batch_norm,
                dropout=dropout
            )
        else:  # attention
            self.fine_encoder = MultiHeadGraphAttention(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                out_channels=hidden_channels,
                num_layers=fine_layers,
                heads=heads,
                edge_channels=7,  # [delta_pos, dist, delta_normal]
                dropout=dropout,
                batch_norm=batch_norm
            )
        
        # Coarse-level encoder
        if encoder_type == 'edgeconv':
            coarse_hidden_list = [coarse_hidden] * coarse_layers
            self.coarse_encoder = EdgeConvEncoder(
                in_channels=hidden_channels,
                hidden_channels=coarse_hidden_list,
                out_channels=coarse_hidden,
                k=k_coarse,
                dynamic=True,
                batch_norm=batch_norm,
                dropout=dropout
            )
        else:  # attention
            self.coarse_encoder = MultiHeadGraphAttention(
                in_channels=hidden_channels,
                hidden_channels=coarse_hidden,
                out_channels=coarse_hidden,
                num_layers=coarse_layers,
                heads=heads,
                edge_channels=7,
                dropout=dropout,
                batch_norm=batch_norm
            )
        
        # Final projection to output dimension
        self.fine_projection = nn.Sequential(
            nn.Linear(hidden_channels, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        self.coarse_projection = nn.Sequential(
            nn.Linear(coarse_hidden, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True)
        )
        
        # Global embedding projection
        self.global_projection = nn.Sequential(
            nn.Linear(out_channels * 2, out_channels),
            nn.BatchNorm1d(out_channels) if batch_norm else nn.Identity(),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels)
        )
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through hierarchical encoder.
        
        Args:
            fine_data: Fine-level PyG Data object
            coarse_data: Coarse-level PyG Data object
            assignments: Mapping from fine to coarse nodes
            batch: Batch indices for global pooling
            
        Returns:
            Dictionary with:
                - 'fine_embeddings': Per-point embeddings (N, D)
                - 'coarse_embeddings': Superpoint embeddings (M, D)
                - 'global_embedding': Global embedding (B, D) or (1, D)
        """
        # Fine-level encoding
        if self.encoder_type == 'edgeconv':
            fine_embeddings = self.fine_encoder(
                fine_data.x,
                edge_index=fine_data.edge_index,
                batch=batch,
                pos=fine_data.pos
            )
        else:
            fine_embeddings = self.fine_encoder(
                fine_data.x,
                fine_data.edge_index,
                fine_data.edge_attr
            )
        
        fine_embeddings = self.fine_projection(fine_embeddings)
        
        # Coarse-level encoding (if provided)
        coarse_embeddings = None
        if coarse_data is not None:
            # Pool fine embeddings to coarse nodes if assignments provided
            if assignments is not None:
                pooled_features = self._pool_to_coarse(
                    fine_embeddings, 
                    assignments,
                    coarse_data.pos.shape[0]
                )
                coarse_input = pooled_features
            else:
                coarse_input = coarse_data.x
            
            if self.encoder_type == 'edgeconv':
                # Create batch tensor for coarse level if needed
                coarse_batch = None
                if batch is not None:
                    # Map batch from fine to coarse
                    coarse_batch = self._map_batch_to_coarse(batch, assignments, coarse_data.pos.shape[0])
                
                coarse_embeddings = self.coarse_encoder(
                    coarse_input,
                    edge_index=coarse_data.edge_index,
                    batch=coarse_batch,
                    pos=coarse_data.pos
                )
            else:
                coarse_embeddings = self.coarse_encoder(
                    coarse_input,
                    coarse_data.edge_index,
                    coarse_data.edge_attr
                )
            
            coarse_embeddings = self.coarse_projection(coarse_embeddings)
        
        # Global embedding
        global_embedding = self._compute_global_embedding(
            fine_embeddings, coarse_embeddings, batch
        )
        
        return {
            'fine_embeddings': fine_embeddings,
            'coarse_embeddings': coarse_embeddings,
            'global_embedding': global_embedding
        }
    
    def _pool_to_coarse(
        self,
        fine_embeddings: torch.Tensor,
        assignments: torch.Tensor,
        num_coarse: int
    ) -> torch.Tensor:
        """Pool fine-level embeddings to coarse-level nodes."""
        device = fine_embeddings.device
        feature_dim = fine_embeddings.shape[1]
        
        pooled = torch.zeros(num_coarse, feature_dim, device=device)
        counts = torch.zeros(num_coarse, 1, device=device)
        
        # Scatter add for mean pooling
        pooled.scatter_add_(
            0, 
            assignments.unsqueeze(1).expand(-1, feature_dim),
            fine_embeddings
        )
        counts.scatter_add_(
            0,
            assignments.unsqueeze(1),
            torch.ones(len(assignments), 1, device=device)
        )
        
        pooled = pooled / (counts + 1e-8)
        return pooled
    
    def _map_batch_to_coarse(
        self,
        batch: torch.Tensor,
        assignments: torch.Tensor,
        num_coarse: int
    ) -> torch.Tensor:
        """Map batch indices from fine to coarse level."""
        device = batch.device
        coarse_batch = torch.zeros(num_coarse, dtype=torch.long, device=device)
        
        # For each coarse node, take the batch index of its first assigned fine node
        for i in range(num_coarse):
            mask = assignments == i
            if mask.any():
                coarse_batch[i] = batch[mask][0]
        
        return coarse_batch
    
    def _compute_global_embedding(
        self,
        fine_embeddings: torch.Tensor,
        coarse_embeddings: Optional[torch.Tensor],
        batch: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Compute global embedding from fine and coarse features."""
        # Global pooling of fine embeddings
        if batch is not None:
            fine_global = global_mean_pool(fine_embeddings, batch)
        else:
            fine_global = fine_embeddings.mean(dim=0, keepdim=True)
        
        # Combine with coarse if available
        if coarse_embeddings is not None:
            if batch is not None:
                # Need to create coarse batch - assume same structure
                coarse_global = coarse_embeddings.mean(dim=0, keepdim=True)
            else:
                coarse_global = coarse_embeddings.mean(dim=0, keepdim=True)
            
            # Match dimensions
            if fine_global.shape[0] != coarse_global.shape[0]:
                coarse_global = coarse_global.expand(fine_global.shape[0], -1)
            
            combined = torch.cat([fine_global, coarse_global], dim=-1)
            global_embedding = self.global_projection(combined)
        else:
            # Only fine-level
            global_embedding = fine_global
        
        return global_embedding
    
    def encode_fine(self, fine_data: Data, batch: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Encode only fine-level features (for faster inference on patches)."""
        if self.encoder_type == 'edgeconv':
            fine_embeddings = self.fine_encoder(
                fine_data.x,
                edge_index=fine_data.edge_index,
                batch=batch,
                pos=fine_data.pos
            )
        else:
            fine_embeddings = self.fine_encoder(
                fine_data.x,
                fine_data.edge_index,
                fine_data.edge_attr
            )
        
        return self.fine_projection(fine_embeddings)


class LightweightEncoder(nn.Module):
    """
    Lightweight encoder variant for faster inference.
    
    Uses fewer layers and smaller dimensions for deployment scenarios.
    """
    
    def __init__(
        self,
        in_channels: int = 8,
        hidden_channels: int = 64,
        out_channels: int = 128,
        num_layers: int = 2,
        k: int = 16
    ):
        super().__init__()
        
        self.layers = nn.ModuleList()
        channels = [in_channels] + [hidden_channels] * num_layers
        
        for i in range(num_layers):
            self.layers.append(
                EdgeConvBlock(
                    channels[i], 
                    channels[i + 1],
                    batch_norm=True,
                    dropout=0.0
                )
            )
        
        self.output = nn.Linear(hidden_channels, out_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index)
        
        return self.output(x)
