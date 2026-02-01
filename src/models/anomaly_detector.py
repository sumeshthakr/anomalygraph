"""
Anomaly Detector Module

Combines the hierarchical encoder with anomaly scoring mechanisms:
1. Memory bank-based density estimation for local anomaly detection
2. Mahalanobis distance for global anomaly detection
3. Score fusion for final anomaly scores
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from typing import Optional, Dict, Tuple
import numpy as np

from .encoder import HierarchicalGraphEncoder


class AnomalyDetector(nn.Module):
    """
    Complete anomaly detection system combining encoder and scoring.
    
    Outputs:
    1. Per-point anomaly heatmap
    2. Object-level anomaly score
    
    Args:
        encoder: HierarchicalGraphEncoder instance (created if None)
        embedding_dim: Embedding dimension
        memory_bank_size: Number of prototypes in memory bank
        local_weight: Weight for local anomaly score (default: 0.7)
        global_weight: Weight for global anomaly score (default: 0.3)
        top_k_percent: Percentage of top scores for object-level score
        temperature: Temperature for score normalization
    """
    
    def __init__(
        self,
        encoder: Optional[HierarchicalGraphEncoder] = None,
        in_channels: int = 8,
        embedding_dim: int = 256,
        memory_bank_size: int = 10000,
        local_weight: float = 0.7,
        global_weight: float = 0.3,
        top_k_percent: float = 0.05,
        temperature: float = 0.1,
        **encoder_kwargs
    ):
        super().__init__()
        
        # Create encoder if not provided
        if encoder is None:
            self.encoder = HierarchicalGraphEncoder(
                in_channels=in_channels,
                out_channels=embedding_dim,
                **encoder_kwargs
            )
        else:
            self.encoder = encoder
            embedding_dim = encoder.out_channels
        
        self.embedding_dim = embedding_dim
        self.memory_bank_size = memory_bank_size
        self.local_weight = local_weight
        self.global_weight = global_weight
        self.top_k_percent = top_k_percent
        self.temperature = temperature
        
        # Memory bank (will be populated during calibration)
        self.register_buffer('memory_bank', None)
        self.register_buffer('global_mean', None)
        self.register_buffer('global_cov_inv', None)
        
        # Learnable projection for better anomaly discrimination
        self.anomaly_projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Score normalization parameters
        self.register_buffer('score_mean', torch.tensor(0.0))
        self.register_buffer('score_std', torch.tensor(1.0))
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None,
        return_embeddings: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for anomaly detection.
        
        Args:
            fine_data: Fine-level graph data
            coarse_data: Coarse-level graph data
            assignments: Fine to coarse node mapping
            batch: Batch indices
            return_embeddings: Whether to return raw embeddings
            
        Returns:
            Dictionary with:
                - 'point_scores': Per-point anomaly scores (N,)
                - 'object_score': Object-level anomaly score (scalar)
                - 'heatmap': Normalized per-point heatmap (N,)
                - 'embeddings': (optional) Raw embeddings dict
        """
        # Get embeddings from encoder
        embeddings = self.encoder(fine_data, coarse_data, assignments, batch)
        
        # Project embeddings for anomaly detection
        fine_embeddings = self.anomaly_projection(embeddings['fine_embeddings'])
        global_embedding = embeddings['global_embedding']
        
        # Compute local anomaly scores
        local_scores = self._compute_local_scores(fine_embeddings)
        
        # Compute global anomaly score
        global_score = self._compute_global_score(global_embedding)
        
        # Fuse scores for per-point scores
        point_scores = self._fuse_scores(local_scores, global_score)
        
        # Compute object-level score
        object_score = self._compute_object_score(point_scores)
        
        # Normalize to heatmap
        heatmap = self._normalize_to_heatmap(point_scores)
        
        result = {
            'point_scores': point_scores,
            'object_score': object_score,
            'heatmap': heatmap,
            'local_scores': local_scores,
            'global_score': global_score
        }
        
        if return_embeddings:
            result['embeddings'] = embeddings
        
        return result
    
    def _compute_local_scores(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute local anomaly scores using memory bank distance.
        
        Args:
            embeddings: Point embeddings (N, D)
            
        Returns:
            Local anomaly scores (N,)
        """
        if self.memory_bank is None:
            # No memory bank calibrated, return zeros
            return torch.zeros(embeddings.shape[0], device=embeddings.device)
        
        # Compute distance to nearest memory bank prototypes
        # Using efficient batch computation
        distances = self._compute_memory_distances(embeddings)
        
        # Take minimum distance (or average of k-nearest)
        k = min(10, self.memory_bank.shape[0])
        topk_distances, _ = torch.topk(distances, k, dim=1, largest=False)
        local_scores = topk_distances.mean(dim=1)
        
        return local_scores
    
    def _compute_memory_distances(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances to memory bank."""
        # Normalize embeddings
        embeddings_norm = F.normalize(embeddings, dim=1)
        memory_norm = F.normalize(self.memory_bank, dim=1)
        
        # Cosine distance
        similarity = torch.mm(embeddings_norm, memory_norm.t())
        distances = 1 - similarity
        
        return distances
    
    def _compute_global_score(self, global_embedding: torch.Tensor) -> torch.Tensor:
        """
        Compute global anomaly score using Mahalanobis distance.
        
        Args:
            global_embedding: Global embedding (B, D) or (1, D)
            
        Returns:
            Global anomaly score (scalar or B,)
        """
        if self.global_mean is None or self.global_cov_inv is None:
            return torch.tensor(0.0, device=global_embedding.device)
        
        # Mahalanobis distance
        diff = global_embedding - self.global_mean
        
        # (x - μ)ᵀ Σ⁻¹ (x - μ)
        mahal = torch.mm(torch.mm(diff, self.global_cov_inv), diff.t())
        
        if global_embedding.shape[0] == 1:
            return mahal.squeeze()
        else:
            return mahal.diag()
    
    def _fuse_scores(
        self,
        local_scores: torch.Tensor,
        global_score: torch.Tensor
    ) -> torch.Tensor:
        """
        Fuse local and global scores.
        
        S_i = local_weight * S_local,i + global_weight * S_global
        """
        # Normalize local scores
        if local_scores.numel() > 0 and local_scores.std() > 0:
            local_normalized = (local_scores - local_scores.mean()) / (local_scores.std() + 1e-8)
        else:
            local_normalized = local_scores
        
        # Normalize global score
        if isinstance(global_score, torch.Tensor) and global_score.numel() > 0:
            global_normalized = global_score / (global_score + 1e-8)
        else:
            global_normalized = 0.0
        
        # Fuse
        fused = self.local_weight * local_normalized + self.global_weight * global_normalized
        
        return fused
    
    def _compute_object_score(self, point_scores: torch.Tensor) -> torch.Tensor:
        """
        Compute object-level anomaly score.
        
        Takes mean of top k% highest point scores.
        """
        n_points = point_scores.shape[0]
        k = max(1, int(n_points * self.top_k_percent))
        
        top_scores, _ = torch.topk(point_scores, k)
        return top_scores.mean()
    
    def _normalize_to_heatmap(self, scores: torch.Tensor) -> torch.Tensor:
        """Normalize scores to [0, 1] heatmap."""
        if scores.numel() == 0:
            return scores
        
        min_val = scores.min()
        max_val = scores.max()
        
        if max_val - min_val > 1e-8:
            heatmap = (scores - min_val) / (max_val - min_val)
        else:
            heatmap = torch.zeros_like(scores)
        
        return heatmap
    
    def calibrate(
        self,
        normal_embeddings: torch.Tensor,
        normal_global_embeddings: torch.Tensor,
        n_prototypes: Optional[int] = None
    ):
        """
        Calibrate the detector using normal sample embeddings.
        
        Args:
            normal_embeddings: Local embeddings from normal samples (N_total, D)
            normal_global_embeddings: Global embeddings from normal samples (M, D)
            n_prototypes: Number of memory bank prototypes (default: memory_bank_size)
        """
        n_prototypes = n_prototypes or self.memory_bank_size
        
        # Project embeddings
        with torch.no_grad():
            projected = self.anomaly_projection(normal_embeddings)
        
        # Create memory bank via k-means clustering
        self._create_memory_bank(projected, n_prototypes)
        
        # Fit Gaussian to global embeddings
        self._fit_global_gaussian(normal_global_embeddings)
        
        # Compute score statistics for normalization
        with torch.no_grad():
            local_scores = self._compute_local_scores(projected)
            self.score_mean = local_scores.mean()
            self.score_std = local_scores.std()
    
    def _create_memory_bank(self, embeddings: torch.Tensor, n_prototypes: int):
        """Create memory bank using k-means clustering."""
        n_samples = embeddings.shape[0]
        n_prototypes = min(n_prototypes, n_samples)
        
        # Use k-means from sklearn or simple random sampling
        try:
            from sklearn.cluster import MiniBatchKMeans
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_prototypes,
                batch_size=min(1024, n_samples),
                n_init=3,
                max_iter=100
            )
            kmeans.fit(embeddings.cpu().numpy())
            prototypes = torch.from_numpy(kmeans.cluster_centers_).to(embeddings.device)
            
        except ImportError:
            # Fallback: random sampling
            indices = torch.randperm(n_samples)[:n_prototypes]
            prototypes = embeddings[indices].clone()
        
        self.memory_bank = prototypes
    
    def _fit_global_gaussian(self, global_embeddings: torch.Tensor):
        """Fit Gaussian distribution to global embeddings."""
        # Compute mean
        self.global_mean = global_embeddings.mean(dim=0, keepdim=True)
        
        # Compute covariance
        centered = global_embeddings - self.global_mean
        cov = torch.mm(centered.t(), centered) / (global_embeddings.shape[0] - 1)
        
        # Regularize and invert
        cov = cov + 0.01 * torch.eye(cov.shape[0], device=cov.device)
        self.global_cov_inv = torch.linalg.inv(cov)
    
    def get_threshold(self, percentile: float = 99.5) -> float:
        """
        Get anomaly threshold based on calibration data.
        
        Should be called after calibrate() with normal samples.
        
        Args:
            percentile: Percentile of normal scores to use as threshold
            
        Returns:
            Anomaly detection threshold
        """
        # This would be computed during calibration
        threshold = self.score_mean + 2.5 * self.score_std
        return threshold.item()


class EnsembleAnomalyDetector(nn.Module):
    """
    Ensemble of anomaly detectors for improved robustness.
    
    Combines multiple detectors with different configurations.
    """
    
    def __init__(
        self,
        detectors: list,
        weights: Optional[list] = None
    ):
        super().__init__()
        
        self.detectors = nn.ModuleList(detectors)
        
        if weights is None:
            weights = [1.0 / len(detectors)] * len(detectors)
        self.weights = weights
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None,
        batch: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Forward pass through all detectors and combine results."""
        all_point_scores = []
        all_object_scores = []
        
        for detector in self.detectors:
            result = detector(fine_data, coarse_data, assignments, batch)
            all_point_scores.append(result['point_scores'])
            all_object_scores.append(result['object_score'])
        
        # Weighted average
        point_scores = torch.zeros_like(all_point_scores[0])
        object_score = torch.tensor(0.0, device=point_scores.device)
        
        for w, ps, os in zip(self.weights, all_point_scores, all_object_scores):
            point_scores += w * ps
            object_score += w * os
        
        # Normalize to heatmap
        heatmap = (point_scores - point_scores.min()) / (point_scores.max() - point_scores.min() + 1e-8)
        
        return {
            'point_scores': point_scores,
            'object_score': object_score,
            'heatmap': heatmap
        }
    
    def calibrate(self, normal_embeddings: torch.Tensor, normal_global_embeddings: torch.Tensor):
        """Calibrate all detectors."""
        for detector in self.detectors:
            detector.calibrate(normal_embeddings, normal_global_embeddings)
