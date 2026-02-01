"""
Training Module for Anomaly Detection

Implements the complete training pipeline including:
1. Self-supervised pretraining
2. Normal feature modeling
3. Anomaly detection training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch_geometric.data import Data, Batch
from typing import Optional, Dict, List, Callable, Tuple
import numpy as np
from tqdm import tqdm
import os
import json
from pathlib import Path


class AnomalyTrainer:
    """
    Complete training pipeline for graph-based anomaly detection.
    
    Handles:
    1. Self-supervised pretraining (optional)
    2. Normal sample calibration
    3. Anomaly detection inference
    
    Args:
        model: The anomaly detector model
        device: Training device ('cuda' or 'cpu')
        learning_rate: Learning rate for optimization
        weight_decay: Weight decay for regularization
        save_dir: Directory for saving checkpoints
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        save_dir: str = './checkpoints'
    ):
        self.model = model.to(device)
        self.device = device
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.training_history = []
    
    def pretrain(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        pretraining_module: Optional[nn.Module] = None,
        epochs: int = 100,
        warmup_epochs: int = 10,
        log_interval: int = 10,
        save_interval: int = 20,
        early_stopping_patience: int = 20
    ) -> Dict[str, List[float]]:
        """
        Self-supervised pretraining of the encoder.
        
        Args:
            train_loader: DataLoader for training data
            val_loader: Optional validation DataLoader
            pretraining_module: Pretraining objective (MaskedPatchModeling, etc.)
            epochs: Number of pretraining epochs
            warmup_epochs: Number of warmup epochs
            log_interval: Logging frequency
            save_interval: Checkpoint saving frequency
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with training history
        """
        from .pretraining import CombinedPretraining
        
        # Create pretraining module if not provided
        if pretraining_module is None:
            pretraining_module = CombinedPretraining(
                self.model.encoder
            ).to(self.device)
        
        # Setup optimizer with warmup
        optimizer = optim.AdamW(
            pretraining_module.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=self.learning_rate * 0.01
        )
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rate': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        print(f"Starting pretraining for {epochs} epochs...")
        
        for epoch in range(epochs):
            self.current_epoch = epoch
            
            # Warmup learning rate
            if epoch < warmup_epochs:
                warmup_lr = self.learning_rate * (epoch + 1) / warmup_epochs
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warmup_lr
            
            # Training
            train_loss = self._pretrain_epoch(
                pretraining_module, 
                train_loader, 
                optimizer
            )
            history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss = self._validate_pretrain(pretraining_module, val_loader)
                history['val_loss'].append(val_loss)
                
                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self.save_checkpoint('pretrain_best.pth')
                else:
                    patience_counter += 1
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            # Learning rate
            history['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Step scheduler after warmup
            if epoch >= warmup_epochs:
                scheduler.step()
            
            # Logging
            if (epoch + 1) % log_interval == 0:
                log_msg = f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss:.4f}"
                if val_loader is not None:
                    log_msg += f" - Val Loss: {history['val_loss'][-1]:.4f}"
                print(log_msg)
            
            # Save checkpoint
            if (epoch + 1) % save_interval == 0:
                self.save_checkpoint(f'pretrain_epoch_{epoch+1}.pth')
        
        # Load best model
        if val_loader is not None and (self.save_dir / 'pretrain_best.pth').exists():
            self.load_checkpoint('pretrain_best.pth')
        
        return history
    
    def _pretrain_epoch(
        self,
        pretraining_module: nn.Module,
        dataloader: DataLoader,
        optimizer: optim.Optimizer
    ) -> float:
        """Single pretraining epoch."""
        pretraining_module.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in tqdm(dataloader, desc="Pretraining", leave=False):
            # Move data to device
            fine_data = batch['fine_graph'].to(self.device)
            coarse_data = batch.get('coarse_graph')
            if coarse_data is not None:
                coarse_data = coarse_data.to(self.device)
            assignments = batch.get('assignments')
            if assignments is not None:
                assignments = assignments.to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            result = pretraining_module(fine_data, coarse_data, assignments)
            loss = result['loss']
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pretraining_module.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def _validate_pretrain(
        self,
        pretraining_module: nn.Module,
        dataloader: DataLoader
    ) -> float:
        """Validation for pretraining."""
        pretraining_module.eval()
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                fine_data = batch['fine_graph'].to(self.device)
                coarse_data = batch.get('coarse_graph')
                if coarse_data is not None:
                    coarse_data = coarse_data.to(self.device)
                assignments = batch.get('assignments')
                if assignments is not None:
                    assignments = assignments.to(self.device)
                
                result = pretraining_module(fine_data, coarse_data, assignments)
                total_loss += result['loss'].item()
                n_batches += 1
        
        return total_loss / max(n_batches, 1)
    
    def calibrate(
        self,
        normal_loader: DataLoader,
        n_prototypes: int = 10000
    ):
        """
        Calibrate the anomaly detector using normal samples.
        
        Collects embeddings from all normal samples and builds
        the memory bank and global Gaussian model.
        
        Args:
            normal_loader: DataLoader with only normal samples
            n_prototypes: Number of memory bank prototypes
        """
        print("Calibrating anomaly detector with normal samples...")
        
        self.model.eval()
        all_embeddings = []
        all_global_embeddings = []
        
        with torch.no_grad():
            for batch in tqdm(normal_loader, desc="Collecting embeddings"):
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
                
                embeddings = result['embeddings']
                all_embeddings.append(embeddings['fine_embeddings'].cpu())
                all_global_embeddings.append(embeddings['global_embedding'].cpu())
        
        # Concatenate all embeddings
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_global_embeddings = torch.cat(all_global_embeddings, dim=0)
        
        print(f"Collected {len(all_embeddings)} point embeddings")
        print(f"Collected {len(all_global_embeddings)} global embeddings")
        
        # Calibrate the model
        self.model.calibrate(
            all_embeddings.to(self.device),
            all_global_embeddings.to(self.device),
            n_prototypes=n_prototypes
        )
        
        print("Calibration complete!")
    
    def evaluate(
        self,
        test_loader: DataLoader,
        compute_per_point: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate the anomaly detector on test data.
        
        Args:
            test_loader: DataLoader with test samples
            compute_per_point: Whether to compute per-point metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        from ..utils.metrics import compute_auroc, compute_pro, compute_fpr_at_tpr
        
        self.model.eval()
        
        all_object_scores = []
        all_object_labels = []
        all_point_scores = []
        all_point_labels = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                fine_data = batch['fine_graph'].to(self.device)
                coarse_data = batch.get('coarse_graph')
                if coarse_data is not None:
                    coarse_data = coarse_data.to(self.device)
                assignments = batch.get('assignments')
                if assignments is not None:
                    assignments = assignments.to(self.device)
                
                labels = batch['label']
                
                # Get predictions
                result = self.model(fine_data, coarse_data, assignments)
                
                all_object_scores.append(result['object_score'].cpu())
                all_object_labels.append(labels)
                
                if compute_per_point:
                    all_point_scores.append(result['point_scores'].cpu())
                    # Assuming per-point labels are available
                    if 'point_labels' in batch:
                        all_point_labels.append(batch['point_labels'])
        
        # Compute object-level metrics
        object_scores = torch.cat(all_object_scores) if len(all_object_scores) > 0 else torch.tensor([])
        object_labels = torch.cat(all_object_labels) if len(all_object_labels) > 0 else torch.tensor([])
        
        if len(object_scores) > 0:
            object_scores = object_scores.numpy()
            object_labels = object_labels.numpy()
        
        metrics = {}
        
        if len(object_scores) > 0 and len(np.unique(object_labels)) > 1:
            metrics['object_auroc'] = compute_auroc(object_labels, object_scores)
            metrics['object_fpr_95'] = compute_fpr_at_tpr(object_labels, object_scores, tpr=0.95)
        
        # Compute per-point metrics if available
        if compute_per_point and len(all_point_labels) > 0:
            point_scores = torch.cat(all_point_scores).numpy()
            point_labels = torch.cat(all_point_labels).numpy()
            
            if len(np.unique(point_labels)) > 1:
                metrics['point_auroc'] = compute_auroc(point_labels, point_scores)
                metrics['pro'] = compute_pro(point_labels, point_scores)
        
        return metrics
    
    def inference(
        self,
        data: Data,
        coarse_data: Optional[Data] = None,
        assignments: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference on a single sample.
        
        Args:
            data: Fine-level graph data
            coarse_data: Coarse-level graph data
            assignments: Fine to coarse mapping
            
        Returns:
            Dictionary with anomaly scores and heatmap
        """
        self.model.eval()
        
        with torch.no_grad():
            data = data.to(self.device)
            if coarse_data is not None:
                coarse_data = coarse_data.to(self.device)
            if assignments is not None:
                assignments = assignments.to(self.device)
            
            result = self.model(data, coarse_data, assignments)
        
        return {k: v.cpu() for k, v in result.items()}
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'epoch': self.current_epoch,
            'best_metric': self.best_metric,
            'training_history': self.training_history
        }
        torch.save(checkpoint, self.save_dir / filename)
    
    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        # Use weights_only=True for security when loading from potentially untrusted sources
        # Note: This requires the checkpoint to contain only tensors and simple Python types
        try:
            checkpoint = torch.load(self.save_dir / filename, map_location=self.device, weights_only=True)
        except Exception:
            # Fallback for checkpoints with custom objects (e.g., config dicts)
            # Only use this with trusted checkpoint sources
            checkpoint = torch.load(self.save_dir / filename, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.current_epoch = checkpoint.get('epoch', 0)
        self.best_metric = checkpoint.get('best_metric', float('inf'))
        self.training_history = checkpoint.get('training_history', [])


def custom_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate function for DataLoader.
    
    Handles PyG Data objects and regular tensors.
    """
    from torch_geometric.data import Batch
    
    collated = {}
    
    # Batch graph data
    if 'fine_graph' in batch[0]:
        collated['fine_graph'] = Batch.from_data_list(
            [item['fine_graph'] for item in batch]
        )
    
    if 'coarse_graph' in batch[0] and batch[0]['coarse_graph'] is not None:
        collated['coarse_graph'] = Batch.from_data_list(
            [item['coarse_graph'] for item in batch]
        )
    
    # Stack tensors
    for key in ['label', 'points', 'features', 'normals', 'curvature', 'roughness']:
        if key in batch[0]:
            if isinstance(batch[0][key], torch.Tensor):
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = torch.tensor([item[key] for item in batch])
    
    # Handle assignments (can't easily batch these)
    if 'assignments' in batch[0]:
        collated['assignments'] = [item['assignments'] for item in batch]
    
    # Keep metadata
    for key in ['category', 'anomaly_type', 'file_path']:
        if key in batch[0]:
            collated[key] = [item[key] for item in batch]
    
    return collated


def create_data_loaders(
    train_dataset,
    val_dataset=None,
    test_dataset=None,
    batch_size: int = 4,
    num_workers: int = 4
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """
    Create DataLoaders for training, validation, and testing.
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    val_loader = None
    if val_dataset is not None:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=1,  # Usually process one at a time for metrics
            shuffle=False,
            num_workers=num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    
    return train_loader, val_loader, test_loader
