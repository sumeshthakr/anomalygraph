"""
Dataset Classes for Anomaly-ShapeNet and Point Cloud Data

This module provides PyTorch dataset implementations for:
1. Anomaly-ShapeNet dataset
2. Generic point cloud dataset
"""

import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Tuple, Callable
import glob
from pathlib import Path

from .preprocessing import PointCloudPreprocessor
from .graph_construction import GraphBuilder


class AnomalyShapeNetDataset(Dataset):
    """
    Dataset class for Anomaly-ShapeNet dataset.
    
    The Anomaly-ShapeNet dataset contains 3D point clouds with synthetic anomalies
    for training and evaluating 3D anomaly detection methods.
    
    Dataset structure expected:
        data_root/
            ├── category1/
            │   ├── train/
            │   │   └── good/
            │   │       └── *.npy or *.ply
            │   └── test/
            │       ├── good/
            │       │   └── *.npy or *.ply
            │       └── anomaly_type/
            │           └── *.npy or *.ply
            ├── category2/
            │   └── ...
            └── ...
    
    Args:
        data_root: Root directory of the dataset
        split: 'train' or 'test'
        categories: List of categories to include (None for all)
        transform: Optional transform to apply to point clouds
        preprocessor: PointCloudPreprocessor instance (creates default if None)
        graph_builder: GraphBuilder instance (creates default if None)
        return_graphs: Whether to return graph structures (default: True)
        point_format: File format ('npy', 'ply', 'pcd', 'txt', 'xyz')
    """
    
    # Known anomaly types in Anomaly-ShapeNet
    ANOMALY_TYPES = [
        'bulge', 'dent', 'hole', 'scratch', 'crack',
        'missing', 'broken', 'bump', 'concavity'
    ]
    
    def __init__(
        self,
        data_root: str,
        split: str = 'train',
        categories: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        preprocessor: Optional[PointCloudPreprocessor] = None,
        graph_builder: Optional[GraphBuilder] = None,
        return_graphs: bool = True,
        point_format: str = 'npy',
        cache_preprocessed: bool = False
    ):
        self.data_root = Path(data_root)
        self.split = split
        self.transform = transform
        self.preprocessor = preprocessor or PointCloudPreprocessor()
        self.graph_builder = graph_builder or GraphBuilder()
        self.return_graphs = return_graphs
        self.point_format = point_format
        self.cache_preprocessed = cache_preprocessed
        
        # Cache for preprocessed data
        self._cache = {}
        
        # Discover categories
        if categories is None:
            self.categories = self._discover_categories()
        else:
            self.categories = categories
        
        # Build sample list
        self.samples = self._build_sample_list()
        
        print(f"Loaded {len(self.samples)} samples from {len(self.categories)} categories")
    
    def _discover_categories(self) -> List[str]:
        """Discover all category folders in the dataset."""
        categories = []
        for item in self.data_root.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                # Check if it has train/test structure
                if (item / 'train').exists() or (item / 'test').exists():
                    categories.append(item.name)
        return sorted(categories)
    
    def _build_sample_list(self) -> List[Dict]:
        """Build list of all samples with metadata."""
        samples = []
        
        for category in self.categories:
            category_path = self.data_root / category / self.split
            
            if not category_path.exists():
                continue
            
            # Iterate through all subfolders (good, anomaly types)
            for label_folder in category_path.iterdir():
                if not label_folder.is_dir():
                    continue
                
                label_name = label_folder.name
                is_anomaly = label_name.lower() != 'good'
                
                # Find all point cloud files
                pattern = f"*.{self.point_format}"
                for file_path in label_folder.glob(pattern):
                    samples.append({
                        'path': str(file_path),
                        'category': category,
                        'label': label_name,
                        'is_anomaly': is_anomaly,
                        'anomaly_type': label_name if is_anomaly else None
                    })
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get a sample from the dataset.
        
        Returns:
            Dictionary containing:
                - 'points': Processed point cloud (N, 3)
                - 'features': Node features (N, 8)
                - 'normals': Surface normals (N, 3)
                - 'label': 0 for normal, 1 for anomaly
                - 'category': Category name
                - 'anomaly_type': Type of anomaly (if any)
                - 'fine_graph': Fine-level PyG Data (if return_graphs)
                - 'coarse_graph': Coarse-level PyG Data (if return_graphs)
        """
        sample_info = self.samples[idx]
        
        # Check cache
        if self.cache_preprocessed and idx in self._cache:
            return self._cache[idx]
        
        # Load point cloud
        points = self._load_point_cloud(sample_info['path'])
        
        # Apply optional transform
        if self.transform is not None:
            points = self.transform(points)
        
        # Preprocess
        processed = self.preprocessor.process(points)
        
        # Build result dictionary
        result = {
            'points': torch.from_numpy(processed['points']),
            'features': torch.from_numpy(processed['features']),
            'normals': torch.from_numpy(processed['normals']),
            'curvature': torch.from_numpy(processed['curvature']),
            'roughness': torch.from_numpy(processed['roughness']),
            'label': 1 if sample_info['is_anomaly'] else 0,
            'category': sample_info['category'],
            'anomaly_type': sample_info['anomaly_type'],
            'file_path': sample_info['path']
        }
        
        # Build graphs if requested
        if self.return_graphs:
            graphs = self.graph_builder.build_hierarchical_graph(
                result['points'],
                result['features'],
                result['normals']
            )
            result['fine_graph'] = graphs['fine']
            result['coarse_graph'] = graphs['coarse']
            result['assignments'] = graphs['assignments']
        
        # Cache if enabled
        if self.cache_preprocessed:
            self._cache[idx] = result
        
        return result
    
    def _load_point_cloud(self, path: str) -> np.ndarray:
        """Load point cloud from various file formats."""
        ext = Path(path).suffix.lower()
        
        if ext == '.npy':
            return np.load(path).astype(np.float32)
        
        elif ext == '.npz':
            data = np.load(path)
            # Try common keys
            for key in ['points', 'point_cloud', 'pts', 'xyz', 'arr_0']:
                if key in data:
                    return data[key].astype(np.float32)
            # Return first array
            return data[list(data.keys())[0]].astype(np.float32)
        
        elif ext == '.ply':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(path)
                return np.asarray(pcd.points).astype(np.float32)
            except ImportError:
                # Fallback: simple PLY parser
                return self._parse_ply(path)
        
        elif ext == '.pcd':
            try:
                import open3d as o3d
                pcd = o3d.io.read_point_cloud(path)
                return np.asarray(pcd.points).astype(np.float32)
            except ImportError:
                raise ImportError("Open3D required for PCD format")
        
        elif ext in ['.txt', '.xyz']:
            return np.loadtxt(path, dtype=np.float32)[:, :3]
        
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def _parse_ply(self, path: str) -> np.ndarray:
        """Simple PLY parser without external dependencies."""
        points = []
        header_ended = False
        vertex_count = 0
        
        with open(path, 'rb') as f:
            for line in f:
                line = line.decode('ascii', errors='ignore').strip()
                
                if not header_ended:
                    if line.startswith('element vertex'):
                        vertex_count = int(line.split()[-1])
                    elif line == 'end_header':
                        header_ended = True
                else:
                    if len(points) < vertex_count:
                        values = line.split()
                        if len(values) >= 3:
                            points.append([float(values[0]), float(values[1]), float(values[2])])
        
        return np.array(points, dtype=np.float32)
    
    def get_category_samples(self, category: str) -> List[int]:
        """Get indices of all samples from a specific category."""
        return [i for i, s in enumerate(self.samples) if s['category'] == category]
    
    def get_normal_samples(self) -> List[int]:
        """Get indices of all normal (non-anomaly) samples."""
        return [i for i, s in enumerate(self.samples) if not s['is_anomaly']]
    
    def get_anomaly_samples(self) -> List[int]:
        """Get indices of all anomaly samples."""
        return [i for i, s in enumerate(self.samples) if s['is_anomaly']]


class PointCloudDataset(Dataset):
    """
    Generic point cloud dataset for arbitrary point cloud files.
    
    Supports loading from:
    - Directory of point cloud files
    - List of file paths
    - NumPy array of point clouds
    
    Args:
        data_source: Directory path, list of file paths, or numpy array
        labels: Optional labels for each sample
        preprocessor: PointCloudPreprocessor instance
        graph_builder: GraphBuilder instance
        return_graphs: Whether to return graph structures
    """
    
    def __init__(
        self,
        data_source,
        labels: Optional[np.ndarray] = None,
        preprocessor: Optional[PointCloudPreprocessor] = None,
        graph_builder: Optional[GraphBuilder] = None,
        return_graphs: bool = True
    ):
        self.preprocessor = preprocessor or PointCloudPreprocessor()
        self.graph_builder = graph_builder or GraphBuilder()
        self.return_graphs = return_graphs
        
        # Handle different data source types
        if isinstance(data_source, (str, Path)):
            self.file_paths = self._find_files(data_source)
            self.point_clouds = None
        elif isinstance(data_source, list):
            if all(isinstance(p, (str, Path)) for p in data_source):
                self.file_paths = data_source
                self.point_clouds = None
            else:
                self.file_paths = None
                self.point_clouds = data_source
        elif isinstance(data_source, np.ndarray):
            self.file_paths = None
            self.point_clouds = [data_source[i] for i in range(len(data_source))]
        else:
            raise ValueError("Unsupported data source type")
        
        self.labels = labels
        
        n_samples = len(self.file_paths) if self.file_paths else len(self.point_clouds)
        print(f"Loaded {n_samples} point clouds")
    
    def _find_files(self, directory: str) -> List[str]:
        """Find all point cloud files in a directory."""
        extensions = ['*.npy', '*.npz', '*.ply', '*.pcd', '*.txt', '*.xyz']
        files = []
        for ext in extensions:
            files.extend(glob.glob(os.path.join(directory, '**', ext), recursive=True))
        return sorted(files)
    
    def __len__(self) -> int:
        if self.file_paths:
            return len(self.file_paths)
        return len(self.point_clouds)
    
    def __getitem__(self, idx: int) -> Dict:
        # Load or retrieve point cloud
        if self.file_paths:
            points = self._load_file(self.file_paths[idx])
        else:
            points = self.point_clouds[idx]
        
        # Preprocess
        processed = self.preprocessor.process(points)
        
        result = {
            'points': torch.from_numpy(processed['points']),
            'features': torch.from_numpy(processed['features']),
            'normals': torch.from_numpy(processed['normals']),
            'idx': idx
        }
        
        if self.labels is not None:
            result['label'] = self.labels[idx]
        
        if self.return_graphs:
            graphs = self.graph_builder.build_hierarchical_graph(
                result['points'],
                result['features'],
                result['normals']
            )
            result['fine_graph'] = graphs['fine']
            result['coarse_graph'] = graphs['coarse']
            result['assignments'] = graphs['assignments']
        
        return result
    
    def _load_file(self, path: str) -> np.ndarray:
        """Load point cloud from file."""
        ext = Path(path).suffix.lower()
        
        if ext == '.npy':
            return np.load(path).astype(np.float32)
        elif ext == '.npz':
            data = np.load(path)
            return data[list(data.keys())[0]].astype(np.float32)
        elif ext in ['.ply', '.pcd']:
            import open3d as o3d
            pcd = o3d.io.read_point_cloud(path)
            return np.asarray(pcd.points).astype(np.float32)
        elif ext in ['.txt', '.xyz']:
            return np.loadtxt(path, dtype=np.float32)[:, :3]
        else:
            raise ValueError(f"Unsupported format: {ext}")


def create_cross_category_splits(
    dataset: AnomalyShapeNetDataset,
    test_categories: List[str],
    val_ratio: float = 0.1
) -> Tuple[List[int], List[int], List[int]]:
    """
    Create train/val/test splits for cross-category evaluation.
    
    Training categories are used for pretraining, test categories
    are held out for evaluation.
    
    Args:
        dataset: AnomalyShapeNetDataset instance
        test_categories: Categories to hold out for testing
        val_ratio: Fraction of training data to use for validation
        
    Returns:
        Tuple of (train_indices, val_indices, test_indices)
    """
    train_categories = [c for c in dataset.categories if c not in test_categories]
    
    train_indices = []
    for cat in train_categories:
        train_indices.extend(dataset.get_category_samples(cat))
    
    # Split training into train/val
    np.random.shuffle(train_indices)
    n_val = int(len(train_indices) * val_ratio)
    val_indices = train_indices[:n_val]
    train_indices = train_indices[n_val:]
    
    # Test indices
    test_indices = []
    for cat in test_categories:
        test_indices.extend(dataset.get_category_samples(cat))
    
    return train_indices, val_indices, test_indices
