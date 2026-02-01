"""
Self-Supervised Pretraining Modules

Implements:
1. Masked Patch Modeling: Predict masked regions
2. Contrastive Learning: Learn invariant representations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import knn_graph
from typing import Optional, Tuple, Dict
import numpy as np
import random


class MaskedPatchModeling(nn.Module):
    """
    Masked Patch Modeling for self-supervised pretraining.
    
    Masks random graph regions and trains the encoder to predict
    the masked coordinates and normals.
    
    Args:
        encoder: The graph encoder to pretrain
        mask_ratio: Fraction of points to mask (default: 0.4)
        patch_size: Number of points per patch for masking
        pred_channels: Number of channels to predict (coords + normals = 6)
        hidden_dim: Hidden dimension for prediction head
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        mask_ratio: float = 0.4,
        patch_size: int = 32,
        pred_channels: int = 6,  # xyz + normals
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        
        embedding_dim = encoder.out_channels
        
        # Prediction head for masked regions
        self.predictor = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, pred_channels)
        )
        
        # Learnable mask token
        self.mask_token = nn.Parameter(torch.randn(1, encoder.out_channels) * 0.02)
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with masked patch modeling.
        
        Args:
            fine_data: Fine-level graph data
            coarse_data: Coarse-level graph data
            assignments: Fine to coarse mapping
            
        Returns:
            Dictionary with loss and predictions
        """
        n_points = fine_data.x.shape[0]
        device = fine_data.x.device
        
        # Create patch-based mask
        mask, patch_centers = self._create_patch_mask(fine_data.pos, n_points)
        
        # Store original features for loss computation
        original_pos = fine_data.pos.clone()
        original_normals = fine_data.normals.clone() if hasattr(fine_data, 'normals') else None
        
        # Mask input features
        masked_data = self._apply_mask(fine_data, mask)
        
        # Encode masked input
        embeddings = self.encoder(masked_data, coarse_data, assignments)
        fine_embeddings = embeddings['fine_embeddings']
        
        # Predict masked regions
        predictions = self.predictor(fine_embeddings[mask])
        
        # Compute targets
        targets_pos = original_pos[mask]
        if original_normals is not None:
            targets = torch.cat([targets_pos, original_normals[mask]], dim=1)
        else:
            targets = targets_pos
        
        # Compute loss
        loss = F.mse_loss(predictions, targets)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'mask': mask
        }
    
    def _create_patch_mask(
        self,
        positions: torch.Tensor,
        n_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create patch-based mask by selecting random patch centers.
        
        Returns:
            Tuple of (mask tensor, patch center indices)
        """
        device = positions.device
        
        # Number of patches to mask
        n_patches = int(n_points * self.mask_ratio / self.patch_size)
        n_patches = max(1, n_patches)
        
        # Randomly select patch centers
        patch_centers = torch.randperm(n_points, device=device)[:n_patches]
        
        # Build kNN graph for patch expansion
        k = min(self.patch_size, n_points - 1)
        edge_index = knn_graph(positions, k=k, loop=True)
        
        # Expand patches from centers
        mask = torch.zeros(n_points, dtype=torch.bool, device=device)
        
        for center in patch_centers:
            # Find neighbors of this center
            neighbors = edge_index[1][edge_index[0] == center]
            mask[neighbors] = True
            mask[center] = True
        
        return mask, patch_centers
    
    def _apply_mask(self, data: Data, mask: torch.Tensor) -> Data:
        """Apply mask by replacing masked features with mask token."""
        masked_data = data.clone()
        
        # Replace masked point features with zeros or learned token
        masked_data.x = data.x.clone()
        masked_data.x[mask] = 0.0  # Zero out masked features
        
        return masked_data


class ContrastiveLearning(nn.Module):
    """
    Contrastive Learning for self-supervised pretraining.
    
    Learns representations that are invariant to augmentations
    (jitter, dropout, scaling, partial crop).
    
    Args:
        encoder: The graph encoder to pretrain
        projection_dim: Dimension of projection head output
        temperature: Temperature for contrastive loss
        hidden_dim: Hidden dimension for projection head
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        projection_dim: int = 128,
        temperature: float = 0.07,
        hidden_dim: int = 256
    ):
        super().__init__()
        
        self.encoder = encoder
        self.temperature = temperature
        
        embedding_dim = encoder.out_channels
        
        # Projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )
        
        # Augmentation parameters
        self.jitter_std = 0.01
        self.dropout_ratio = 0.1
        self.scale_range = (0.8, 1.2)
        self.crop_ratio = 0.2
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass with contrastive learning.
        
        Args:
            fine_data: Fine-level graph data
            coarse_data: Coarse-level graph data
            assignments: Fine to coarse mapping
            
        Returns:
            Dictionary with contrastive loss and embeddings
        """
        # Create two augmented views
        view1_data = self._augment(fine_data)
        view2_data = self._augment(fine_data)
        
        # Encode both views
        emb1 = self.encoder(view1_data, coarse_data, assignments)
        emb2 = self.encoder(view2_data, coarse_data, assignments)
        
        # Get global embeddings
        z1 = emb1['global_embedding']
        z2 = emb2['global_embedding']
        
        # Project
        p1 = self.projector(z1)
        p2 = self.projector(z2)
        
        # Normalize
        p1 = F.normalize(p1, dim=1)
        p2 = F.normalize(p2, dim=1)
        
        # Compute contrastive loss
        loss = self._nt_xent_loss(p1, p2)
        
        return {
            'loss': loss,
            'embeddings1': z1,
            'embeddings2': z2,
            'projections1': p1,
            'projections2': p2
        }
    
    def _augment(self, data: Data) -> Data:
        """Apply random augmentations to point cloud."""
        aug_data = data.clone()
        device = data.pos.device
        n_points = data.pos.shape[0]
        
        # Random jitter
        if random.random() < 0.8:
            jitter = torch.randn_like(data.pos) * self.jitter_std
            aug_data.pos = data.pos + jitter
        
        # Random scaling
        if random.random() < 0.8:
            scale = random.uniform(*self.scale_range)
            aug_data.pos = aug_data.pos * scale
        
        # Random dropout (remove points)
        if random.random() < 0.5 and n_points > 100:
            n_keep = int(n_points * (1 - self.dropout_ratio))
            keep_indices = torch.randperm(n_points, device=device)[:n_keep]
            
            aug_data.pos = aug_data.pos[keep_indices]
            aug_data.x = aug_data.x[keep_indices]
            
            if hasattr(aug_data, 'normals') and aug_data.normals is not None:
                aug_data.normals = aug_data.normals[keep_indices]
            
            # Rebuild graph
            aug_data.edge_index = knn_graph(aug_data.pos, k=24, loop=False)
        
        # Random rotation (small angles)
        if random.random() < 0.5:
            angle = random.uniform(-np.pi / 12, np.pi / 12)  # ±15 degrees
            axis = random.choice(['x', 'y', 'z'])
            aug_data.pos = self._rotate(aug_data.pos, angle, axis)
            
            if hasattr(aug_data, 'normals') and aug_data.normals is not None:
                aug_data.normals = self._rotate(aug_data.normals, angle, axis)
        
        return aug_data
    
    def _rotate(self, points: torch.Tensor, angle: float, axis: str) -> torch.Tensor:
        """Rotate points around specified axis."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if axis == 'x':
            R = torch.tensor([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ], dtype=points.dtype, device=points.device)
        elif axis == 'y':
            R = torch.tensor([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ], dtype=points.dtype, device=points.device)
        else:  # z
            R = torch.tensor([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ], dtype=points.dtype, device=points.device)
        
        return torch.mm(points, R.t())
    
    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Normalized Temperature-scaled Cross Entropy Loss (NT-Xent).
        
        InfoNCE loss for contrastive learning.
        """
        batch_size = z1.shape[0]
        
        if batch_size == 1:
            # Can't compute contrastive loss with single sample
            return torch.tensor(0.0, device=z1.device)
        
        # Concatenate representations
        representations = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        similarity = torch.mm(representations, representations.t())  # (2B, 2B)
        similarity = similarity / self.temperature
        
        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.arange(batch_size, device=z1.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z1.device)
        similarity.masked_fill_(mask, float('-inf'))
        
        # Cross entropy loss
        loss = F.cross_entropy(similarity, labels)
        
        return loss


class CombinedPretraining(nn.Module):
    """
    Combined pretraining objective using both masked modeling and contrastive learning.
    
    Args:
        encoder: The graph encoder to pretrain
        mask_weight: Weight for masked patch modeling loss
        contrastive_weight: Weight for contrastive loss
        **kwargs: Additional arguments for individual modules
    """
    
    def __init__(
        self,
        encoder: nn.Module,
        mask_weight: float = 1.0,
        contrastive_weight: float = 0.5,
        **kwargs
    ):
        super().__init__()
        
        self.masked_modeling = MaskedPatchModeling(encoder, **kwargs)
        self.contrastive = ContrastiveLearning(encoder, **kwargs)
        
        self.mask_weight = mask_weight
        self.contrastive_weight = contrastive_weight
    
    def forward(
        self,
        fine_data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """Combined pretraining forward pass."""
        # Masked modeling loss
        mask_result = self.masked_modeling(fine_data, coarse_data, assignments)
        
        # Contrastive loss
        contrastive_result = self.contrastive(fine_data, coarse_data, assignments)
        
        # Combined loss
        total_loss = (
            self.mask_weight * mask_result['loss'] +
            self.contrastive_weight * contrastive_result['loss']
        )
        
        return {
            'loss': total_loss,
            'mask_loss': mask_result['loss'],
            'contrastive_loss': contrastive_result['loss']
        }


class PointCloudAugmentation:
    """
    Comprehensive point cloud augmentation for training.
    
    Includes:
    - Jitter: Add random noise
    - Dropout: Remove random points
    - Scaling: Random uniform scaling
    - Rotation: Random rotation
    - Cropping: Random partial crop
    - Translation: Random translation
    """
    
    def __init__(
        self,
        jitter_std: float = 0.01,
        dropout_ratio: float = 0.1,
        scale_range: Tuple[float, float] = (0.8, 1.2),
        rotation_range: float = np.pi / 4,
        translation_range: float = 0.1,
        crop_ratio: float = 0.2
    ):
        self.jitter_std = jitter_std
        self.dropout_ratio = dropout_ratio
        self.scale_range = scale_range
        self.rotation_range = rotation_range
        self.translation_range = translation_range
        self.crop_ratio = crop_ratio
    
    def __call__(self, points: np.ndarray) -> np.ndarray:
        """Apply random augmentations."""
        augmented = points.copy()
        
        # Jitter
        if random.random() < 0.8:
            augmented += np.random.randn(*points.shape) * self.jitter_std
        
        # Scale
        if random.random() < 0.8:
            scale = random.uniform(*self.scale_range)
            augmented *= scale
        
        # Rotation
        if random.random() < 0.5:
            angle = random.uniform(-self.rotation_range, self.rotation_range)
            axis = random.choice(['x', 'y', 'z'])
            augmented = self._rotate_np(augmented, angle, axis)
        
        # Translation
        if random.random() < 0.5:
            translation = np.random.uniform(
                -self.translation_range, 
                self.translation_range, 
                size=3
            )
            augmented += translation
        
        # Dropout
        if random.random() < 0.3 and len(points) > 100:
            n_keep = int(len(points) * (1 - self.dropout_ratio))
            indices = np.random.choice(len(points), n_keep, replace=False)
            augmented = augmented[indices]
        
        return augmented
    
    def _rotate_np(self, points: np.ndarray, angle: float, axis: str) -> np.ndarray:
        """Rotate points using numpy."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        
        if axis == 'x':
            R = np.array([
                [1, 0, 0],
                [0, cos_a, -sin_a],
                [0, sin_a, cos_a]
            ])
        elif axis == 'y':
            R = np.array([
                [cos_a, 0, sin_a],
                [0, 1, 0],
                [-sin_a, 0, cos_a]
            ])
        else:
            R = np.array([
                [cos_a, -sin_a, 0],
                [sin_a, cos_a, 0],
                [0, 0, 1]
            ])
        
        return points @ R.T
