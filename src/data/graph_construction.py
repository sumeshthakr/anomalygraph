"""
Graph Construction Module

Builds multi-scale geometric graphs for anomaly detection:
1. Fine Graph: For micro-defect detection (scratches, pits, small holes)
2. Coarse Graph: For macro-defect detection (bulges, dents, missing parts)
"""

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph, fps
from typing import Tuple, Optional, Dict
from scipy.spatial import cKDTree


class GraphBuilder:
    """
    Multi-scale graph construction for 3D point clouds.
    
    Builds hierarchical graphs:
    - Fine graph: All points connected via kNN
    - Coarse graph: Superpoints (FPS sampled) with pooled features
    
    Args:
        fine_k: Number of neighbors for fine graph (default: 24)
        coarse_k: Number of neighbors for coarse graph (default: 16)
        num_superpoints: Number of superpoints for coarse graph (default: 2048)
        include_edge_features: Whether to compute edge features (default: True)
    """
    
    def __init__(
        self,
        fine_k: int = 24,
        coarse_k: int = 16,
        num_superpoints: int = 2048,
        include_edge_features: bool = True
    ):
        self.fine_k = fine_k
        self.coarse_k = coarse_k
        self.num_superpoints = num_superpoints
        self.include_edge_features = include_edge_features
    
    def build_fine_graph(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> Data:
        """
        Build fine-level graph for micro-defect detection.
        
        Args:
            points: Point coordinates (N, 3)
            features: Node features (N, F)
            normals: Surface normals (N, 3) for edge features
            
        Returns:
            PyG Data object with:
                - x: Node features
                - pos: Point positions
                - edge_index: Graph connectivity
                - edge_attr: Edge features (if enabled)
        """
        if not torch.is_tensor(points):
            points = torch.from_numpy(points).float()
        if not torch.is_tensor(features):
            features = torch.from_numpy(features).float()
        if normals is not None and not torch.is_tensor(normals):
            normals = torch.from_numpy(normals).float()
        
        n_points = points.shape[0]
        
        # Build kNN graph
        edge_index = knn_graph(points, k=self.fine_k, loop=False)
        
        # Compute edge features if requested
        edge_attr = None
        if self.include_edge_features:
            edge_attr = self._compute_edge_features(
                points, normals, edge_index
            )
        
        data = Data(
            x=features,
            pos=points,
            edge_index=edge_index,
            edge_attr=edge_attr
        )
        
        if normals is not None:
            data.normals = normals
        
        return data
    
    def build_coarse_graph(
        self,
        fine_data: Data,
        fine_embeddings: Optional[torch.Tensor] = None
    ) -> Tuple[Data, torch.Tensor]:
        """
        Build coarse-level graph for macro-defect detection.
        
        Uses Farthest Point Sampling (FPS) to select superpoints,
        then pools fine-level features to create coarse node features.
        
        Args:
            fine_data: Fine-level PyG Data object
            fine_embeddings: Optional fine-level embeddings to pool
            
        Returns:
            Tuple of:
                - Coarse-level PyG Data object
                - Assignment indices mapping fine to coarse nodes
        """
        points = fine_data.pos
        n_points = points.shape[0]
        
        # Limit superpoints to available points
        num_superpoints = min(self.num_superpoints, n_points)
        
        # Farthest Point Sampling
        batch = torch.zeros(n_points, dtype=torch.long, device=points.device)
        fps_indices = fps(points, batch, ratio=num_superpoints / n_points)
        
        if len(fps_indices) < num_superpoints:
            # Pad with random samples if FPS returns fewer points
            remaining = num_superpoints - len(fps_indices)
            available = list(set(range(n_points)) - set(fps_indices.tolist()))
            if len(available) > 0:
                extra = np.random.choice(available, min(remaining, len(available)), replace=False)
                fps_indices = torch.cat([fps_indices, torch.tensor(extra, device=points.device)])
        
        superpoint_pos = points[fps_indices]
        
        # Assign each fine point to nearest superpoint
        assignments = self._assign_to_superpoints(points, superpoint_pos)
        
        # Pool features to superpoints
        if fine_embeddings is not None:
            features_to_pool = fine_embeddings
        else:
            features_to_pool = fine_data.x
        
        superpoint_features = self._pool_features(
            features_to_pool, assignments, num_superpoints
        )
        
        # Pool normals if available
        superpoint_normals = None
        if hasattr(fine_data, 'normals') and fine_data.normals is not None:
            superpoint_normals = self._pool_features(
                fine_data.normals, assignments, num_superpoints
            )
            # Re-normalize pooled normals
            superpoint_normals = superpoint_normals / (
                superpoint_normals.norm(dim=1, keepdim=True) + 1e-8
            )
        
        # Build coarse kNN graph
        coarse_edge_index = knn_graph(
            superpoint_pos, k=self.coarse_k, loop=False
        )
        
        # Compute edge features for coarse graph
        coarse_edge_attr = None
        if self.include_edge_features:
            coarse_edge_attr = self._compute_edge_features(
                superpoint_pos, superpoint_normals, coarse_edge_index
            )
        
        coarse_data = Data(
            x=superpoint_features,
            pos=superpoint_pos,
            edge_index=coarse_edge_index,
            edge_attr=coarse_edge_attr
        )
        
        if superpoint_normals is not None:
            coarse_data.normals = superpoint_normals
        
        # Store FPS indices for reference
        coarse_data.fps_indices = fps_indices
        
        return coarse_data, assignments
    
    def build_hierarchical_graph(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        normals: Optional[torch.Tensor] = None
    ) -> Dict[str, Data]:
        """
        Build complete hierarchical graph structure.
        
        Args:
            points: Point coordinates (N, 3)
            features: Node features (N, F)
            normals: Surface normals (N, 3)
            
        Returns:
            Dictionary with:
                - 'fine': Fine-level graph
                - 'coarse': Coarse-level graph
                - 'assignments': Fine to coarse mapping
        """
        fine_graph = self.build_fine_graph(points, features, normals)
        coarse_graph, assignments = self.build_coarse_graph(fine_graph)
        
        return {
            'fine': fine_graph,
            'coarse': coarse_graph,
            'assignments': assignments
        }
    
    def _compute_edge_features(
        self,
        points: torch.Tensor,
        normals: Optional[torch.Tensor],
        edge_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute edge features for graph edges.
        
        Edge features include:
        - Relative position (3D)
        - Distance (1D)
        - Normal difference (3D) if normals provided
        
        Returns:
            Edge features tensor (E, 4 or 7)
        """
        src, dst = edge_index
        
        # Relative position
        delta_pos = points[dst] - points[src]  # (E, 3)
        
        # Distance
        distance = torch.norm(delta_pos, dim=1, keepdim=True)  # (E, 1)
        
        edge_features = [delta_pos, distance]
        
        # Normal difference if available
        if normals is not None:
            delta_normal = normals[dst] - normals[src]  # (E, 3)
            edge_features.append(delta_normal)
        
        return torch.cat(edge_features, dim=1)
    
    def _assign_to_superpoints(
        self,
        points: torch.Tensor,
        superpoints: torch.Tensor
    ) -> torch.Tensor:
        """Assign each point to its nearest superpoint."""
        # Use cdist for efficient distance computation
        if points.is_cuda:
            dists = torch.cdist(points, superpoints)
            assignments = dists.argmin(dim=1)
        else:
            # CPU: use scipy for better performance
            points_np = points.numpy()
            superpoints_np = superpoints.numpy()
            tree = cKDTree(superpoints_np)
            _, assignments_np = tree.query(points_np)
            assignments = torch.from_numpy(assignments_np)
        
        return assignments
    
    def _pool_features(
        self,
        features: torch.Tensor,
        assignments: torch.Tensor,
        num_superpoints: int
    ) -> torch.Tensor:
        """Pool features from fine to coarse level using mean aggregation."""
        device = features.device
        feature_dim = features.shape[1]
        
        pooled = torch.zeros(num_superpoints, feature_dim, device=device)
        counts = torch.zeros(num_superpoints, 1, device=device)
        
        # Scatter add for pooling
        pooled.scatter_add_(0, assignments.unsqueeze(1).expand(-1, feature_dim), features)
        counts.scatter_add_(0, assignments.unsqueeze(1), torch.ones(len(assignments), 1, device=device))
        
        # Average pooling
        pooled = pooled / (counts + 1e-8)
        
        return pooled


class DynamicGraphBuilder(GraphBuilder):
    """
    Dynamic graph construction that recomputes edges based on learned features.
    
    Useful for the EdgeConv-style dynamic graph updates during forward pass.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def recompute_edges(
        self,
        embeddings: torch.Tensor,
        k: Optional[int] = None
    ) -> torch.Tensor:
        """
        Recompute graph edges based on embedding similarity.
        
        Args:
            embeddings: Node embeddings (N, D)
            k: Number of neighbors (default: use self.fine_k)
            
        Returns:
            New edge_index tensor
        """
        k = k or self.fine_k
        return knn_graph(embeddings, k=k, loop=False)


def collate_graphs(graph_list: list) -> Data:
    """
    Collate multiple graphs into a single batched graph.
    
    Args:
        graph_list: List of PyG Data objects
        
    Returns:
        Batched PyG Data object
    """
    from torch_geometric.data import Batch
    return Batch.from_data_list(graph_list)
