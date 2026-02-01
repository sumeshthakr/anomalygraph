"""
Memory Bank for Normal Feature Density Estimation

Implements efficient memory bank storage and querying for
anomaly detection based on distance to normal prototypes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class MemoryBank(nn.Module):
    """
    Memory Bank for storing and querying normal feature prototypes.
    
    Uses k-means clustering to compress embeddings into prototypes,
    then computes anomaly scores based on distance to nearest prototypes.
    
    Args:
        feature_dim: Dimension of feature embeddings
        bank_size: Maximum number of prototypes
        distance_metric: 'cosine' or 'euclidean'
        k_neighbors: Number of nearest neighbors for scoring
    """
    
    def __init__(
        self,
        feature_dim: int,
        bank_size: int = 10000,
        distance_metric: str = 'cosine',
        k_neighbors: int = 10
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.bank_size = bank_size
        self.distance_metric = distance_metric
        self.k_neighbors = k_neighbors
        
        # Memory bank storage
        self.register_buffer('bank', None)
        self.register_buffer('bank_size_current', torch.tensor(0))
        
        # Statistics for normalization
        self.register_buffer('mean', None)
        self.register_buffer('std', None)
    
    def update(self, embeddings: torch.Tensor, compress: bool = True):
        """
        Update memory bank with new embeddings.
        
        Args:
            embeddings: New embeddings to add (N, D)
            compress: Whether to compress using k-means
        """
        device = embeddings.device
        
        # Initialize or update statistics
        if self.mean is None:
            self.mean = embeddings.mean(dim=0)
            self.std = embeddings.std(dim=0) + 1e-8
        else:
            # Running update
            n_old = self.bank_size_current.item()
            n_new = embeddings.shape[0]
            total = n_old + n_new
            
            new_mean = embeddings.mean(dim=0)
            new_std = embeddings.std(dim=0) + 1e-8
            
            self.mean = (self.mean * n_old + new_mean * n_new) / total
            self.std = (self.std * n_old + new_std * n_new) / total
        
        # Normalize embeddings
        normalized = (embeddings - self.mean) / self.std
        
        if compress and normalized.shape[0] > self.bank_size:
            # Compress using k-means
            prototypes = self._kmeans_compress(normalized)
        else:
            prototypes = normalized
        
        # Update bank
        if self.bank is None:
            self.bank = prototypes
        else:
            self.bank = torch.cat([self.bank, prototypes], dim=0)
            
            # Limit size
            if self.bank.shape[0] > self.bank_size:
                indices = torch.randperm(self.bank.shape[0])[:self.bank_size]
                self.bank = self.bank[indices]
        
        self.bank_size_current = torch.tensor(self.bank.shape[0], device=device)
    
    def _kmeans_compress(
        self,
        embeddings: torch.Tensor,
        n_clusters: Optional[int] = None
    ) -> torch.Tensor:
        """Compress embeddings using k-means clustering."""
        n_clusters = n_clusters or self.bank_size
        n_samples = embeddings.shape[0]
        n_clusters = min(n_clusters, n_samples)
        
        try:
            from sklearn.cluster import MiniBatchKMeans
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_clusters,
                batch_size=min(1024, n_samples),
                n_init=3,
                max_iter=100
            )
            kmeans.fit(embeddings.cpu().numpy())
            prototypes = torch.from_numpy(kmeans.cluster_centers_).to(embeddings.device)
            
        except ImportError:
            # Fallback: random sampling
            indices = torch.randperm(n_samples)[:n_clusters]
            prototypes = embeddings[indices].clone()
        
        return prototypes.float()
    
    def query(
        self,
        embeddings: torch.Tensor,
        k: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Query memory bank for nearest neighbor distances.
        
        Args:
            embeddings: Query embeddings (N, D)
            k: Number of neighbors (default: self.k_neighbors)
            
        Returns:
            Tuple of (distances, indices) for k nearest neighbors
        """
        if self.bank is None:
            return torch.zeros(embeddings.shape[0]), torch.zeros(embeddings.shape[0], dtype=torch.long)
        
        k = k or self.k_neighbors
        k = min(k, self.bank.shape[0])
        
        # Normalize query embeddings
        normalized = (embeddings - self.mean) / self.std
        
        if self.distance_metric == 'cosine':
            distances = self._cosine_distance(normalized, self.bank)
        else:
            distances = self._euclidean_distance(normalized, self.bank)
        
        # Get k nearest
        topk_distances, topk_indices = torch.topk(distances, k, dim=1, largest=False)
        
        return topk_distances, topk_indices
    
    def _cosine_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute cosine distance between x and y."""
        x_norm = F.normalize(x, dim=1)
        y_norm = F.normalize(y, dim=1)
        similarity = torch.mm(x_norm, y_norm.t())
        return 1 - similarity
    
    def _euclidean_distance(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Compute Euclidean distance between x and y."""
        # (x - y)^2 = x^2 + y^2 - 2xy
        x_sq = (x ** 2).sum(dim=1, keepdim=True)
        y_sq = (y ** 2).sum(dim=1, keepdim=True)
        xy = torch.mm(x, y.t())
        distances = x_sq + y_sq.t() - 2 * xy
        return torch.sqrt(torch.clamp(distances, min=1e-8))
    
    def compute_anomaly_scores(
        self,
        embeddings: torch.Tensor,
        aggregation: str = 'mean'
    ) -> torch.Tensor:
        """
        Compute anomaly scores based on distance to memory bank.
        
        Args:
            embeddings: Query embeddings (N, D)
            aggregation: How to aggregate k-NN distances ('mean', 'min', 'max')
            
        Returns:
            Anomaly scores (N,)
        """
        distances, _ = self.query(embeddings)
        
        if aggregation == 'mean':
            scores = distances.mean(dim=1)
        elif aggregation == 'min':
            scores = distances.min(dim=1).values
        elif aggregation == 'max':
            scores = distances.max(dim=1).values
        else:
            scores = distances.mean(dim=1)
        
        return scores
    
    def save(self, path: str):
        """Save memory bank to file."""
        torch.save({
            'bank': self.bank,
            'mean': self.mean,
            'std': self.std,
            'feature_dim': self.feature_dim,
            'bank_size': self.bank_size,
            'distance_metric': self.distance_metric,
            'k_neighbors': self.k_neighbors
        }, path)
    
    def load(self, path: str):
        """Load memory bank from file."""
        # Note: weights_only=True is safer but requires the data to be simple tensors
        # Since we only store tensors here, we use weights_only=True for security
        data = torch.load(path, weights_only=True)
        self.bank = data['bank']
        self.mean = data['mean']
        self.std = data['std']
        self.bank_size_current = torch.tensor(self.bank.shape[0] if self.bank is not None else 0)


class FAISSMemoryBank:
    """
    Memory Bank using FAISS for efficient similarity search.
    
    Provides much faster queries for large memory banks.
    """
    
    def __init__(
        self,
        feature_dim: int,
        use_gpu: bool = False
    ):
        try:
            import faiss
            self.faiss = faiss
        except ImportError:
            raise ImportError("FAISS required for FAISSMemoryBank. Install with: pip install faiss-cpu")
        
        self.feature_dim = feature_dim
        self.use_gpu = use_gpu
        
        # Create index
        self.index = faiss.IndexFlatL2(feature_dim)
        
        if use_gpu:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
        
        self.embeddings = None
    
    def update(self, embeddings: np.ndarray):
        """Add embeddings to the index."""
        embeddings = embeddings.astype(np.float32)
        
        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        self.index.add(embeddings)
    
    def query(
        self,
        embeddings: np.ndarray,
        k: int = 10
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query for k nearest neighbors.
        
        Returns:
            Tuple of (distances, indices)
        """
        embeddings = embeddings.astype(np.float32)
        distances, indices = self.index.search(embeddings, k)
        return distances, indices
    
    def compute_anomaly_scores(
        self,
        embeddings: np.ndarray,
        k: int = 10
    ) -> np.ndarray:
        """Compute anomaly scores as mean distance to k neighbors."""
        distances, _ = self.query(embeddings, k)
        return distances.mean(axis=1)
    
    def reset(self):
        """Reset the index."""
        self.index.reset()
        self.embeddings = None


class HierarchicalMemoryBank(nn.Module):
    """
    Hierarchical memory bank for multi-scale anomaly detection.
    
    Maintains separate banks for fine and coarse level features.
    """
    
    def __init__(
        self,
        fine_feature_dim: int,
        coarse_feature_dim: int,
        fine_bank_size: int = 10000,
        coarse_bank_size: int = 5000
    ):
        super().__init__()
        
        self.fine_bank = MemoryBank(fine_feature_dim, fine_bank_size)
        self.coarse_bank = MemoryBank(coarse_feature_dim, coarse_bank_size)
        
        # Fusion weights
        self.register_buffer('fine_weight', torch.tensor(0.7))
        self.register_buffer('coarse_weight', torch.tensor(0.3))
    
    def update(
        self,
        fine_embeddings: torch.Tensor,
        coarse_embeddings: torch.Tensor
    ):
        """Update both memory banks."""
        self.fine_bank.update(fine_embeddings)
        self.coarse_bank.update(coarse_embeddings)
    
    def compute_anomaly_scores(
        self,
        fine_embeddings: torch.Tensor,
        coarse_embeddings: torch.Tensor,
        fine_to_coarse: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute fused anomaly scores from both levels.
        
        Args:
            fine_embeddings: Fine-level embeddings (N, D1)
            coarse_embeddings: Coarse-level embeddings (M, D2)
            fine_to_coarse: Mapping from fine to coarse indices
            
        Returns:
            Per-point anomaly scores (N,)
        """
        # Fine-level scores
        fine_scores = self.fine_bank.compute_anomaly_scores(fine_embeddings)
        
        # Coarse-level scores
        coarse_scores = self.coarse_bank.compute_anomaly_scores(coarse_embeddings)
        
        # Map coarse scores to fine level if mapping provided
        if fine_to_coarse is not None:
            coarse_scores_mapped = coarse_scores[fine_to_coarse]
        else:
            # Broadcast single coarse score
            coarse_scores_mapped = coarse_scores.mean().expand(fine_scores.shape[0])
        
        # Fuse scores
        fused = self.fine_weight * fine_scores + self.coarse_weight * coarse_scores_mapped
        
        return fused
