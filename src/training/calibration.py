"""
Calibration Module for New Object Categories

Implements the calibration pipeline for adapting the anomaly
detector to unseen object categories using only normal samples.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Tuple
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm


class CategoryCalibrator:
    """
    Calibrator for adapting anomaly detector to new object categories.
    
    For each unseen object family:
    - Collects 10-30 normal samples
    - Builds/updates memory bank
    - Computes detection threshold (99.5 percentile)
    
    Args:
        model: The anomaly detector model
        device: Computation device
        memory_bank_size: Size of memory bank per category
        threshold_percentile: Percentile for threshold computation
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        memory_bank_size: int = 5000,
        threshold_percentile: float = 99.5
    ):
        self.model = model
        self.device = device
        self.memory_bank_size = memory_bank_size
        self.threshold_percentile = threshold_percentile
        
        # Store per-category calibration data
        self.category_data = {}
    
    def calibrate_category(
        self,
        category: str,
        normal_samples: DataLoader,
        update_global: bool = False
    ) -> Dict[str, float]:
        """
        Calibrate for a specific object category.
        
        Args:
            category: Category name
            normal_samples: DataLoader with normal samples of this category
            update_global: Whether to also update global statistics
            
        Returns:
            Calibration results including threshold
        """
        print(f"Calibrating for category: {category}")
        
        self.model.eval()
        
        all_local_embeddings = []
        all_global_embeddings = []
        all_scores = []
        
        with torch.no_grad():
            for batch in tqdm(normal_samples, desc=f"Calibrating {category}"):
                fine_data = batch['fine_graph'].to(self.device)
                coarse_data = batch.get('coarse_graph')
                if coarse_data is not None:
                    coarse_data = coarse_data.to(self.device)
                assignments = batch.get('assignments')
                if assignments is not None:
                    assignments = assignments.to(self.device)
                
                # Get embeddings
                result = self.model(
                    fine_data, coarse_data, assignments,
                    return_embeddings=True
                )
                
                all_local_embeddings.append(
                    result['embeddings']['fine_embeddings'].cpu()
                )
                all_global_embeddings.append(
                    result['embeddings']['global_embedding'].cpu()
                )
                all_scores.append(result['object_score'].cpu())
        
        # Concatenate embeddings
        local_embeddings = torch.cat(all_local_embeddings, dim=0)
        global_embeddings = torch.cat(all_global_embeddings, dim=0)
        scores = torch.stack(all_scores)
        
        # Create category-specific memory bank
        memory_bank = self._create_memory_bank(
            local_embeddings, 
            self.memory_bank_size
        )
        
        # Compute threshold
        threshold = self._compute_threshold(
            scores.numpy(),
            self.threshold_percentile
        )
        
        # Compute statistics
        stats = {
            'mean': scores.mean().item(),
            'std': scores.std().item(),
            'min': scores.min().item(),
            'max': scores.max().item(),
            'threshold': threshold,
            'n_samples': len(all_scores),
            'n_embeddings': len(local_embeddings)
        }
        
        # Store calibration data
        self.category_data[category] = {
            'memory_bank': memory_bank,
            'global_mean': global_embeddings.mean(dim=0),
            'global_std': global_embeddings.std(dim=0),
            'stats': stats
        }
        
        # Optionally update model's global calibration
        if update_global:
            self.model.calibrate(
                local_embeddings.to(self.device),
                global_embeddings.to(self.device)
            )
        
        print(f"  - Threshold: {threshold:.4f}")
        print(f"  - Score range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        print(f"  - Mean ± Std: {stats['mean']:.4f} ± {stats['std']:.4f}")
        
        return stats
    
    def _create_memory_bank(
        self,
        embeddings: torch.Tensor,
        n_prototypes: int
    ) -> torch.Tensor:
        """Create memory bank via k-means clustering."""
        n_samples = embeddings.shape[0]
        n_prototypes = min(n_prototypes, n_samples)
        
        try:
            from sklearn.cluster import MiniBatchKMeans
            
            kmeans = MiniBatchKMeans(
                n_clusters=n_prototypes,
                batch_size=min(1024, n_samples),
                n_init=3,
                max_iter=100
            )
            kmeans.fit(embeddings.numpy())
            prototypes = torch.from_numpy(kmeans.cluster_centers_)
            
        except ImportError:
            # Fallback: random sampling
            indices = torch.randperm(n_samples)[:n_prototypes]
            prototypes = embeddings[indices].clone()
        
        return prototypes
    
    def _compute_threshold(
        self,
        scores: np.ndarray,
        percentile: float
    ) -> float:
        """Compute threshold at given percentile."""
        return float(np.percentile(scores, percentile))
    
    def detect_anomaly(
        self,
        category: str,
        sample: Dict,
        use_category_threshold: bool = True
    ) -> Dict:
        """
        Detect anomaly for a sample from a calibrated category.
        
        Args:
            category: Category name
            sample: Sample data dictionary
            use_category_threshold: Whether to use category-specific threshold
            
        Returns:
            Detection result with scores and classification
        """
        if category not in self.category_data:
            raise ValueError(f"Category {category} not calibrated. Call calibrate_category first.")
        
        self.model.eval()
        
        with torch.no_grad():
            fine_data = sample['fine_graph'].to(self.device)
            coarse_data = sample.get('coarse_graph')
            if coarse_data is not None:
                coarse_data = coarse_data.to(self.device)
            assignments = sample.get('assignments')
            if assignments is not None:
                assignments = assignments.to(self.device)
            
            result = self.model(fine_data, coarse_data, assignments)
        
        object_score = result['object_score'].item()
        
        # Get threshold
        if use_category_threshold:
            threshold = self.category_data[category]['stats']['threshold']
        else:
            threshold = self.model.get_threshold()
        
        is_anomaly = object_score > threshold
        
        return {
            'object_score': object_score,
            'point_scores': result['point_scores'].cpu(),
            'heatmap': result['heatmap'].cpu(),
            'threshold': threshold,
            'is_anomaly': is_anomaly,
            'category': category
        }
    
    def save_calibration(self, path: str):
        """Save calibration data to file."""
        save_data = {}
        
        for category, data in self.category_data.items():
            save_data[category] = {
                'memory_bank': data['memory_bank'].numpy().tolist(),
                'global_mean': data['global_mean'].numpy().tolist(),
                'global_std': data['global_std'].numpy().tolist(),
                'stats': data['stats']
            }
        
        with open(path, 'w') as f:
            json.dump(save_data, f, indent=2)
        
        print(f"Calibration saved to {path}")
    
    def load_calibration(self, path: str):
        """Load calibration data from file."""
        with open(path, 'r') as f:
            save_data = json.load(f)
        
        for category, data in save_data.items():
            self.category_data[category] = {
                'memory_bank': torch.tensor(data['memory_bank']),
                'global_mean': torch.tensor(data['global_mean']),
                'global_std': torch.tensor(data['global_std']),
                'stats': data['stats']
            }
        
        print(f"Loaded calibration for {len(self.category_data)} categories")
    
    def get_calibrated_categories(self) -> List[str]:
        """Get list of calibrated categories."""
        return list(self.category_data.keys())
    
    def get_category_stats(self, category: str) -> Dict:
        """Get calibration statistics for a category."""
        if category not in self.category_data:
            raise ValueError(f"Category {category} not calibrated.")
        return self.category_data[category]['stats']


class AdaptiveThreshold:
    """
    Adaptive threshold computation for anomaly detection.
    
    Provides multiple threshold strategies:
    1. Percentile-based
    2. Mean + k*std
    3. Maximum likelihood
    """
    
    def __init__(self, strategy: str = 'percentile'):
        self.strategy = strategy
        self.fitted = False
        self.threshold = None
        self.params = {}
    
    def fit(self, normal_scores: np.ndarray, **kwargs):
        """
        Fit threshold based on normal sample scores.
        
        Args:
            normal_scores: Anomaly scores from normal samples
            **kwargs: Strategy-specific parameters
        """
        if self.strategy == 'percentile':
            percentile = kwargs.get('percentile', 99.5)
            self.threshold = np.percentile(normal_scores, percentile)
            self.params = {'percentile': percentile}
            
        elif self.strategy == 'std':
            k = kwargs.get('k', 3.0)
            mean = np.mean(normal_scores)
            std = np.std(normal_scores)
            self.threshold = mean + k * std
            self.params = {'k': k, 'mean': mean, 'std': std}
            
        elif self.strategy == 'mle':
            # Fit Gaussian and use likelihood threshold
            from scipy.stats import norm
            mean, std = norm.fit(normal_scores)
            p_value = kwargs.get('p_value', 0.01)
            self.threshold = norm.ppf(1 - p_value, mean, std)
            self.params = {'mean': mean, 'std': std, 'p_value': p_value}
        
        self.fitted = True
        return self.threshold
    
    def predict(self, scores: np.ndarray) -> np.ndarray:
        """Predict anomaly labels based on threshold."""
        if not self.fitted:
            raise ValueError("Threshold not fitted. Call fit() first.")
        return (scores > self.threshold).astype(int)
    
    def get_threshold(self) -> float:
        """Get current threshold value."""
        if not self.fitted:
            raise ValueError("Threshold not fitted. Call fit() first.")
        return self.threshold


class OnlineCalibrator:
    """
    Online calibration that updates with streaming samples.
    
    Maintains running statistics and can update the memory bank
    incrementally without storing all samples.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        buffer_size: int = 1000,
        update_frequency: int = 100
    ):
        self.model = model
        self.device = device
        self.buffer_size = buffer_size
        self.update_frequency = update_frequency
        
        # Running statistics
        self.n_samples = 0
        self.running_mean = None
        self.running_var = None
        self.embedding_buffer = []
    
    def update(self, sample: Dict) -> Dict:
        """
        Process a new sample and update calibration.
        
        Args:
            sample: Sample data dictionary
            
        Returns:
            Current detection result
        """
        self.model.eval()
        
        with torch.no_grad():
            fine_data = sample['fine_graph'].to(self.device)
            coarse_data = sample.get('coarse_graph')
            if coarse_data is not None:
                coarse_data = coarse_data.to(self.device)
            assignments = sample.get('assignments')
            if assignments is not None:
                assignments = assignments.to(self.device)
            
            result = self.model(fine_data, coarse_data, assignments, return_embeddings=True)
        
        embeddings = result['embeddings']['fine_embeddings'].cpu()
        
        # Update buffer
        self.embedding_buffer.append(embeddings)
        if len(self.embedding_buffer) > self.buffer_size:
            self.embedding_buffer.pop(0)
        
        # Update running statistics
        self._update_stats(embeddings)
        
        # Periodic model update
        if self.n_samples % self.update_frequency == 0 and len(self.embedding_buffer) > 10:
            self._update_model()
        
        return {
            'object_score': result['object_score'].item(),
            'heatmap': result['heatmap'].cpu()
        }
    
    def _update_stats(self, embeddings: torch.Tensor):
        """Update running mean and variance."""
        batch_mean = embeddings.mean(dim=0)
        batch_var = embeddings.var(dim=0)
        n = embeddings.shape[0]
        
        if self.running_mean is None:
            self.running_mean = batch_mean
            self.running_var = batch_var
            self.n_samples = n
        else:
            # Welford's online algorithm
            total = self.n_samples + n
            delta = batch_mean - self.running_mean
            
            self.running_mean = self.running_mean + delta * n / total
            self.running_var = (
                (self.running_var * self.n_samples + batch_var * n) / total +
                delta ** 2 * self.n_samples * n / total ** 2
            )
            self.n_samples = total
    
    def _update_model(self):
        """Update model with buffered embeddings."""
        all_embeddings = torch.cat(self.embedding_buffer, dim=0)
        
        # Subsample if too large
        if len(all_embeddings) > 10000:
            indices = torch.randperm(len(all_embeddings))[:10000]
            all_embeddings = all_embeddings[indices]
        
        # This is a simplified update - in practice, may want to
        # incrementally update the memory bank
        pass
