#!/usr/bin/env python3
"""
RunPod Training Script for Graph-Based 3D Point-Cloud Anomaly Detection

This script is designed to run on RunPod GPU instances for training
the anomaly detection model on the Anomaly-ShapeNet dataset.

Usage:
    python train_runpod.py --data_root /path/to/Anomaly-ShapeNet --epochs 100
    
For RunPod:
    1. Upload this script and the src/ folder to your RunPod instance
    2. Install requirements: pip install -r requirements.txt
    3. Run: python scripts/train_runpod.py --config configs/default.yaml
"""

import os
import sys
import argparse
import yaml
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import (
    PointCloudPreprocessor,
    GraphBuilder,
    AnomalyShapeNetDataset,
    PointCloudDataset
)
from src.models import (
    HierarchicalGraphEncoder,
    AnomalyDetector
)
from src.training import (
    AnomalyTrainer,
    CategoryCalibrator
)
from src.training.pretraining import CombinedPretraining
from src.training.trainer import custom_collate_fn
from src.utils.metrics import compute_auroc, compute_pro, CategoryMetrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Graph-Based 3D Anomaly Detection on RunPod'
    )
    
    # Data arguments
    parser.add_argument('--data_root', type=str, default='./data/Anomaly-ShapeNet',
                        help='Path to Anomaly-ShapeNet dataset')
    parser.add_argument('--config', type=str, default='./configs/default.yaml',
                        help='Path to config file')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of pretraining epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Model arguments
    parser.add_argument('--encoder_type', type=str, default='edgeconv',
                        choices=['edgeconv', 'attention'],
                        help='Encoder type')
    parser.add_argument('--embedding_dim', type=int, default=256,
                        help='Embedding dimension')
    
    # Output arguments
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory for checkpoints and logs')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name (default: timestamp)')
    
    # Misc arguments
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--eval_only', action='store_true',
                        help='Only run evaluation')
    parser.add_argument('--test_categories', type=str, nargs='+', default=None,
                        help='Categories to hold out for testing')
    
    return parser.parse_args()


def setup_environment(args):
    """Setup training environment."""
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    if args.device == 'cuda':
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Setup experiment directory
    if args.experiment_name is None:
        args.experiment_name = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    experiment_dir = Path(args.output_dir) / args.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories
    (experiment_dir / 'checkpoints').mkdir(exist_ok=True)
    (experiment_dir / 'logs').mkdir(exist_ok=True)
    (experiment_dir / 'visualizations').mkdir(exist_ok=True)
    
    return device, experiment_dir


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_datasets(args, config, preprocessor, graph_builder):
    """Create training and test datasets."""
    data_root = Path(args.data_root)
    
    if not data_root.exists():
        raise FileNotFoundError(f"Dataset not found at {data_root}")
    
    # Discover categories
    all_categories = []
    for item in data_root.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            if (item / 'train').exists() or (item / 'test').exists():
                all_categories.append(item.name)
    
    print(f"Found {len(all_categories)} categories: {all_categories}")
    
    # Split categories for cross-category evaluation
    if args.test_categories:
        test_categories = args.test_categories
    else:
        # Default: use last category for testing
        test_categories = [all_categories[-1]] if all_categories else []
    
    train_categories = [c for c in all_categories if c not in test_categories]
    
    print(f"Training categories: {train_categories}")
    print(f"Test categories: {test_categories}")
    
    # Create datasets
    train_dataset = AnomalyShapeNetDataset(
        data_root=str(data_root),
        split='train',
        categories=train_categories,
        preprocessor=preprocessor,
        graph_builder=graph_builder,
        return_graphs=True,
        point_format=config['data']['point_format']
    )
    
    test_dataset = AnomalyShapeNetDataset(
        data_root=str(data_root),
        split='test',
        categories=test_categories,
        preprocessor=preprocessor,
        graph_builder=graph_builder,
        return_graphs=True,
        point_format=config['data']['point_format']
    )
    
    return train_dataset, test_dataset, train_categories, test_categories


def create_model(args, config, device):
    """Create the anomaly detection model."""
    encoder = HierarchicalGraphEncoder(
        in_channels=config['model']['in_channels'],
        hidden_channels=config['model']['hidden_channels'],
        coarse_hidden=config['model']['coarse_hidden'],
        out_channels=args.embedding_dim,
        fine_layers=config['model']['fine_layers'],
        coarse_layers=config['model']['coarse_layers'],
        encoder_type=args.encoder_type,
        k_fine=config['graph']['fine_k'],
        k_coarse=config['graph']['coarse_k'],
        heads=config['model']['heads'],
        dropout=config['model']['dropout'],
        batch_norm=config['model']['batch_norm']
    )
    
    model = AnomalyDetector(
        encoder=encoder,
        embedding_dim=args.embedding_dim,
        memory_bank_size=config['anomaly']['memory_bank_size'],
        local_weight=config['anomaly']['local_weight'],
        global_weight=config['anomaly']['global_weight'],
        top_k_percent=config['anomaly']['top_k_percent']
    ).to(device)
    
    return model


def pretrain(model, train_loader, val_loader, args, config, experiment_dir, device):
    """Run self-supervised pretraining."""
    print("\n" + "="*60)
    print("Stage 1: Self-Supervised Pretraining")
    print("="*60)
    
    # Create pretraining module
    pretraining_module = CombinedPretraining(
        encoder=model.encoder,
        mask_weight=1.0,
        contrastive_weight=config['training']['pretrain']['contrastive_weight'],
        mask_ratio=config['training']['pretrain']['mask_ratio'],
        patch_size=config['training']['pretrain']['patch_size'],
        projection_dim=config['training']['pretrain']['projection_dim'],
        temperature=config['training']['pretrain']['temperature']
    ).to(device)
    
    # Create trainer
    trainer = AnomalyTrainer(
        model=model,
        device=device,
        learning_rate=args.lr,
        weight_decay=config['training']['pretrain']['weight_decay'],
        save_dir=str(experiment_dir / 'checkpoints')
    )
    
    # Run pretraining
    history = trainer.pretrain(
        train_loader=train_loader,
        val_loader=val_loader,
        pretraining_module=pretraining_module,
        epochs=args.epochs,
        warmup_epochs=config['training']['pretrain']['warmup_epochs'],
        log_interval=config['logging']['log_interval'],
        save_interval=config['training']['pretrain']['save_interval'],
        early_stopping_patience=config['training']['pretrain']['early_stopping_patience']
    )
    
    # Save training history
    with open(experiment_dir / 'logs' / 'pretrain_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    return trainer, history


def calibrate(model, normal_loader, config, device):
    """Calibrate the anomaly detector with normal samples."""
    print("\n" + "="*60)
    print("Stage 2: Calibration with Normal Samples")
    print("="*60)
    
    model.eval()
    all_embeddings = []
    all_global_embeddings = []
    
    with torch.no_grad():
        for batch in tqdm(normal_loader, desc="Collecting embeddings"):
            fine_data = batch['fine_graph'].to(device)
            coarse_data = batch.get('coarse_graph')
            if coarse_data is not None:
                coarse_data = coarse_data.to(device)
            assignments = batch.get('assignments')
            if assignments is not None:
                assignments = assignments.to(device)
            
            result = model(fine_data, coarse_data, assignments, return_embeddings=True)
            
            all_embeddings.append(result['embeddings']['fine_embeddings'].cpu())
            all_global_embeddings.append(result['embeddings']['global_embedding'].cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_global_embeddings = torch.cat(all_global_embeddings, dim=0)
    
    print(f"Collected {len(all_embeddings)} point embeddings")
    print(f"Collected {len(all_global_embeddings)} global embeddings")
    
    model.calibrate(
        all_embeddings.to(device),
        all_global_embeddings.to(device),
        n_prototypes=config['anomaly']['memory_bank_size']
    )
    
    print("Calibration complete!")


def evaluate(model, test_loader, device):
    """Evaluate the model on test data."""
    print("\n" + "="*60)
    print("Evaluation")
    print("="*60)
    
    model.eval()
    metrics_tracker = CategoryMetrics()
    
    all_object_scores = []
    all_object_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            fine_data = batch['fine_graph'].to(device)
            coarse_data = batch.get('coarse_graph')
            if coarse_data is not None:
                coarse_data = coarse_data.to(device)
            assignments = batch.get('assignments')
            if assignments is not None:
                assignments = assignments.to(device)
            
            result = model(fine_data, coarse_data, assignments)
            
            # Collect predictions
            for i in range(len(batch['label'])):
                category = batch['category'][i] if 'category' in batch else 'unknown'
                label = batch['label'][i].item() if torch.is_tensor(batch['label'][i]) else batch['label'][i]
                score = result['object_score'].item()
                
                metrics_tracker.add(category, label, score)
                all_object_scores.append(score)
                all_object_labels.append(label)
    
    # Compute metrics
    object_scores = np.array(all_object_scores)
    object_labels = np.array(all_object_labels)
    
    if len(np.unique(object_labels)) > 1:
        auroc = compute_auroc(object_labels, object_scores)
        print(f"\nObject-Level AUROC: {auroc:.4f}")
    else:
        auroc = 0.5
        print("\nWarning: Only one class present in test data")
    
    # Print per-category results
    results = metrics_tracker.compute()
    print("\nPer-Category Results:")
    for category, cat_metrics in results['per_category'].items():
        if 'object_auroc' in cat_metrics:
            print(f"  {category}: AUROC = {cat_metrics['object_auroc']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    print("="*60)
    print("Graph-Based 3D Point-Cloud Anomaly Detection - RunPod Training")
    print("="*60)
    print(f"\nArguments: {args}")
    
    # Setup environment
    device, experiment_dir = setup_environment(args)
    print(f"Experiment directory: {experiment_dir}")
    
    # Load config
    config = load_config(args.config)
    
    # Override config with command line arguments
    config['training']['batch_size'] = args.batch_size
    config['training']['num_workers'] = args.num_workers
    
    # Save config to experiment directory
    with open(experiment_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Create preprocessor and graph builder
    preprocessor = PointCloudPreprocessor(
        target_points=config['data']['target_points'],
        outlier_nb_neighbors=config['data']['outlier_nb_neighbors'],
        outlier_std_ratio=config['data']['outlier_std_ratio'],
        normal_knn=config['data']['normal_knn']
    )
    
    graph_builder = GraphBuilder(
        fine_k=config['graph']['fine_k'],
        coarse_k=config['graph']['coarse_k'],
        num_superpoints=config['graph']['num_superpoints'],
        include_edge_features=config['graph']['include_edge_features']
    )
    
    # Create datasets
    print("\n" + "="*60)
    print("Loading Dataset")
    print("="*60)
    
    train_dataset, test_dataset, train_categories, test_categories = create_datasets(
        args, config, preprocessor, graph_builder
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Validation: use subset of training data
    val_indices = list(range(0, len(train_dataset), 10))[:100]  # Every 10th sample
    if val_indices:
        val_dataset = Subset(train_dataset, val_indices)
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True
        )
    else:
        val_loader = None
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=custom_collate_fn,
        pin_memory=True
    )
    
    # Create model
    print("\n" + "="*60)
    print("Creating Model")
    print("="*60)
    
    model = create_model(args, config, device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        # Use weights_only=True for security when loading from potentially untrusted sources
        try:
            checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        except Exception:
            # Fallback for checkpoints with custom objects (only use with trusted sources)
            checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    if not args.eval_only:
        # Stage 1: Pretraining
        trainer, history = pretrain(
            model, train_loader, val_loader,
            args, config, experiment_dir, device
        )
        
        # Stage 2: Calibration
        # Get normal samples for calibration
        normal_indices = train_dataset.get_normal_samples()
        if len(normal_indices) > 0:
            # Use subset for calibration
            calib_indices = normal_indices[:min(100, len(normal_indices))]
            calib_dataset = Subset(train_dataset, calib_indices)
            normal_loader = DataLoader(
                calib_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=args.num_workers,
                collate_fn=custom_collate_fn
            )
            calibrate(model, normal_loader, config, device)
        
        # Save final model
        torch.save({
            'model_state_dict': model.state_dict(),
            'config': config,
            'args': vars(args)
        }, experiment_dir / 'checkpoints' / 'final_model.pth')
    
    # Evaluation
    results = evaluate(model, test_loader, device)
    
    # Save results
    with open(experiment_dir / 'results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Results saved to: {experiment_dir}")
    
    return results


if __name__ == '__main__':
    main()
