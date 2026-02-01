#!/usr/bin/env python3
"""
Inference Script for Anomaly Detection

Run inference on a single point cloud or directory of point clouds.

Usage:
    python inference.py --model checkpoint.pth --input sample.ply --output result.json
    python inference.py --model checkpoint.pth --input ./test_data/ --output ./results/
"""

import os
import sys
import argparse
import json
from pathlib import Path

import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data import PointCloudPreprocessor, GraphBuilder
from src.models import AnomalyDetector, HierarchicalGraphEncoder
from src.utils import visualize_heatmap


def load_model(checkpoint_path, device='cuda'):
    """Load trained model from checkpoint."""
    # Use weights_only=True for security when loading from potentially untrusted sources
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except Exception:
        # Fallback for checkpoints with custom objects (only use with trusted sources)
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})
    
    # Create encoder
    encoder = HierarchicalGraphEncoder(
        in_channels=model_config.get('in_channels', 8),
        hidden_channels=model_config.get('hidden_channels', 128),
        coarse_hidden=model_config.get('coarse_hidden', 256),
        out_channels=model_config.get('out_channels', 256),
        fine_layers=model_config.get('fine_layers', 3),
        coarse_layers=model_config.get('coarse_layers', 2),
        encoder_type=model_config.get('encoder_type', 'edgeconv')
    )
    
    # Create detector
    anomaly_config = config.get('anomaly', {})
    model = AnomalyDetector(
        encoder=encoder,
        embedding_dim=model_config.get('out_channels', 256),
        memory_bank_size=anomaly_config.get('memory_bank_size', 10000),
        local_weight=anomaly_config.get('local_weight', 0.7),
        global_weight=anomaly_config.get('global_weight', 0.3)
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model, config


def load_point_cloud(path):
    """Load point cloud from various formats."""
    path = Path(path)
    ext = path.suffix.lower()
    
    if ext == '.npy':
        return np.load(path).astype(np.float32)
    elif ext == '.npz':
        data = np.load(path)
        for key in ['points', 'point_cloud', 'pts', 'xyz', 'arr_0']:
            if key in data:
                return data[key].astype(np.float32)
        return data[list(data.keys())[0]].astype(np.float32)
    elif ext in ['.ply', '.pcd']:
        import open3d as o3d
        pcd = o3d.io.read_point_cloud(str(path))
        return np.asarray(pcd.points).astype(np.float32)
    elif ext in ['.txt', '.xyz']:
        return np.loadtxt(path, dtype=np.float32)[:, :3]
    else:
        raise ValueError(f"Unsupported format: {ext}")


def run_inference(model, points, preprocessor, graph_builder, device='cuda'):
    """Run inference on a single point cloud."""
    # Preprocess
    processed = preprocessor.process(points)
    
    # Build graphs
    pts = torch.from_numpy(processed['points'])
    features = torch.from_numpy(processed['features'])
    normals = torch.from_numpy(processed['normals'])
    
    graphs = graph_builder.build_hierarchical_graph(pts, features, normals)
    
    # Move to device
    fine_data = graphs['fine'].to(device)
    coarse_data = graphs['coarse'].to(device)
    assignments = graphs['assignments'].to(device)
    
    # Inference
    with torch.no_grad():
        result = model(fine_data, coarse_data, assignments)
    
    return {
        'points': pts.numpy(),
        'object_score': result['object_score'].cpu().item(),
        'point_scores': result['point_scores'].cpu().numpy(),
        'heatmap': result['heatmap'].cpu().numpy(),
        'is_anomaly': result['object_score'].item() > model.get_threshold()
    }


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Inference')
    parser.add_argument('--model', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--input', type=str, required=True,
                        help='Input point cloud file or directory')
    parser.add_argument('--output', type=str, default='./results',
                        help='Output directory or file')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use')
    parser.add_argument('--threshold', type=float, default=None,
                        help='Custom anomaly threshold')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    device = torch.device(args.device)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, device)
    
    # Create preprocessor and graph builder
    data_config = config.get('data', {})
    graph_config = config.get('graph', {})
    
    preprocessor = PointCloudPreprocessor(
        target_points=data_config.get('target_points', 100000),
        normal_knn=data_config.get('normal_knn', 30)
    )
    
    graph_builder = GraphBuilder(
        fine_k=graph_config.get('fine_k', 24),
        coarse_k=graph_config.get('coarse_k', 16),
        num_superpoints=graph_config.get('num_superpoints', 2048)
    )
    
    # Process input
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if input_path.is_file():
        # Single file
        files = [input_path]
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        # Directory
        extensions = ['*.npy', '*.npz', '*.ply', '*.pcd', '*.txt', '*.xyz']
        files = []
        for ext in extensions:
            files.extend(input_path.glob(f'**/{ext}'))
        output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing {len(files)} file(s)...")
    
    results = []
    for file_path in files:
        print(f"\nProcessing: {file_path}")
        
        try:
            # Load point cloud
            points = load_point_cloud(file_path)
            print(f"  Points: {len(points)}")
            
            # Run inference
            result = run_inference(
                model, points, preprocessor, graph_builder, device
            )
            
            print(f"  Object Score: {result['object_score']:.4f}")
            print(f"  Prediction: {'ANOMALY' if result['is_anomaly'] else 'NORMAL'}")
            
            # Save result
            result_data = {
                'file': str(file_path),
                'object_score': result['object_score'],
                'is_anomaly': result['is_anomaly'],
                'n_points': len(result['points'])
            }
            results.append(result_data)
            
            # Visualize if requested
            if args.visualize:
                vis_path = output_path / f"{file_path.stem}_heatmap.png"
                visualize_heatmap(
                    result['points'],
                    result['heatmap'],
                    title=f"Score: {result['object_score']:.4f}",
                    save_path=str(vis_path),
                    show=False
                )
                print(f"  Visualization saved: {vis_path}")
            
            # Save per-point scores
            np.save(
                output_path / f"{file_path.stem}_scores.npy",
                result['point_scores']
            )
            
        except Exception as e:
            print(f"  Error: {e}")
            results.append({
                'file': str(file_path),
                'error': str(e)
            })
    
    # Save summary
    summary_path = output_path / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print(f"Summary: {summary_path}")


if __name__ == '__main__':
    main()
