"""
Point Cloud Preprocessing Pipeline

This module implements the preprocessing pipeline for 3D point clouds:
1. Outlier removal (statistical filtering)
2. Voxel downsampling to fixed resolution
3. Scale normalization (unit bounding box)
4. Per-point feature estimation (normals, curvature, roughness)
"""

import numpy as np
from typing import Tuple, Optional, Dict
import torch


class PointCloudPreprocessor:
    """
    Comprehensive preprocessing pipeline for 3D point clouds.
    
    Prepares point clouds for graph-based anomaly detection by:
    - Removing outliers
    - Downsampling to consistent resolution
    - Normalizing scale
    - Computing geometric features (normals, curvature, roughness)
    
    Args:
        target_points: Target number of points after downsampling (default: 100000)
        voxel_size: Voxel size for downsampling (auto-computed if None)
        outlier_nb_neighbors: Number of neighbors for outlier detection (default: 20)
        outlier_std_ratio: Standard deviation ratio for outlier removal (default: 2.0)
        normal_knn: Number of neighbors for normal estimation (default: 30)
        use_open3d: Whether to use Open3D for processing (default: True)
    """
    
    def __init__(
        self,
        target_points: int = 100000,
        voxel_size: Optional[float] = None,
        outlier_nb_neighbors: int = 20,
        outlier_std_ratio: float = 2.0,
        normal_knn: int = 30,
        use_open3d: bool = True
    ):
        self.target_points = target_points
        self.voxel_size = voxel_size
        self.outlier_nb_neighbors = outlier_nb_neighbors
        self.outlier_std_ratio = outlier_std_ratio
        self.normal_knn = normal_knn
        self.use_open3d = use_open3d
        
    def process(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for a point cloud.
        
        Args:
            points: Input point cloud of shape (N, 3)
            
        Returns:
            Dictionary containing:
                - 'points': Processed points (M, 3)
                - 'normals': Surface normals (M, 3)
                - 'curvature': Curvature values (M,)
                - 'roughness': Roughness values (M,)
                - 'features': Combined feature vector (M, 8)
        """
        if self.use_open3d:
            return self._process_open3d(points)
        else:
            return self._process_numpy(points)
    
    def _process_open3d(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Process using Open3D library for optimal performance."""
        try:
            import open3d as o3d
        except ImportError:
            print("Open3D not available, falling back to numpy implementation")
            return self._process_numpy(points)
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
        
        # Step 1: Remove outliers
        pcd, inlier_indices = pcd.remove_statistical_outlier(
            nb_neighbors=self.outlier_nb_neighbors,
            std_ratio=self.outlier_std_ratio
        )
        
        # Step 2: Voxel downsampling
        if self.voxel_size is not None:
            pcd = pcd.voxel_down_sample(voxel_size=self.voxel_size)
        else:
            # Auto-compute voxel size to achieve target point count
            current_points = len(pcd.points)
            if current_points > self.target_points:
                # Estimate voxel size based on bounding box
                bbox = pcd.get_axis_aligned_bounding_box()
                extent = bbox.get_extent()
                volume = np.prod(extent)
                # Estimate voxel size to achieve target density
                voxel_size = (volume / self.target_points) ** (1/3) * 0.8
                pcd = pcd.voxel_down_sample(voxel_size=voxel_size)
                
                # Iteratively refine if needed
                while len(pcd.points) > self.target_points * 1.1:
                    voxel_size *= 1.1
                    pcd_temp = o3d.geometry.PointCloud()
                    pcd_temp.points = o3d.utility.Vector3dVector(points.astype(np.float64))
                    pcd = pcd_temp.voxel_down_sample(voxel_size=voxel_size)
        
        # Step 3: Scale normalization (unit bounding box)
        points_normalized = np.asarray(pcd.points)
        center = points_normalized.mean(axis=0)
        points_normalized = points_normalized - center
        scale = np.max(np.abs(points_normalized))
        if scale > 0:
            points_normalized = points_normalized / scale
        
        pcd.points = o3d.utility.Vector3dVector(points_normalized)
        
        # Step 4: Estimate normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamKNN(knn=self.normal_knn)
        )
        pcd.orient_normals_consistent_tangent_plane(k=self.normal_knn)
        
        normals = np.asarray(pcd.normals)
        final_points = np.asarray(pcd.points)
        
        # Step 5: Compute curvature and roughness
        curvature, roughness = self._compute_curvature_roughness_o3d(pcd)
        
        # Combine features: [x, y, z, nx, ny, nz, curvature, roughness]
        features = np.concatenate([
            final_points,
            normals,
            curvature.reshape(-1, 1),
            roughness.reshape(-1, 1)
        ], axis=1)
        
        return {
            'points': final_points.astype(np.float32),
            'normals': normals.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'roughness': roughness.astype(np.float32),
            'features': features.astype(np.float32)
        }
    
    def _compute_curvature_roughness_o3d(self, pcd) -> Tuple[np.ndarray, np.ndarray]:
        """Compute curvature and roughness using PCA on local neighborhoods."""
        import open3d as o3d
        
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)
        n_points = len(points)
        
        # Build KD-tree
        pcd_tree = o3d.geometry.KDTreeFlann(pcd)
        
        curvature = np.zeros(n_points, dtype=np.float32)
        roughness = np.zeros(n_points, dtype=np.float32)
        
        for i in range(n_points):
            # Find k nearest neighbors
            _, idx, _ = pcd_tree.search_knn_vector_3d(points[i], self.normal_knn)
            idx = np.asarray(idx)
            
            if len(idx) < 3:
                continue
            
            # Get neighborhood points
            neighbors = points[idx]
            
            # Compute covariance matrix for curvature
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            
            # Eigenvalue decomposition
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)
            
            # Curvature proxy: λ₃ / (λ₁ + λ₂ + λ₃)
            total = eigenvalues.sum()
            if total > 1e-10:
                curvature[i] = eigenvalues[0] / total
            
            # Roughness: normal variance in neighborhood
            neighbor_normals = normals[idx]
            normal_variance = np.var(neighbor_normals, axis=0).sum()
            roughness[i] = normal_variance
        
        # Normalize roughness to [0, 1]
        if roughness.max() > 0:
            roughness = roughness / roughness.max()
            
        return curvature, roughness
    
    def _process_numpy(self, points: np.ndarray) -> Dict[str, np.ndarray]:
        """Fallback processing using only NumPy/SciPy."""
        from scipy.spatial import cKDTree
        
        # Step 1: Statistical outlier removal
        tree = cKDTree(points)
        distances, _ = tree.query(points, k=self.outlier_nb_neighbors + 1)
        mean_distances = distances[:, 1:].mean(axis=1)
        
        mean_dist = mean_distances.mean()
        std_dist = mean_distances.std()
        threshold = mean_dist + self.outlier_std_ratio * std_dist
        
        inlier_mask = mean_distances < threshold
        points = points[inlier_mask]
        
        # Step 2: Random downsampling (simpler than voxel)
        if len(points) > self.target_points:
            indices = np.random.choice(len(points), self.target_points, replace=False)
            points = points[indices]
        
        # Step 3: Scale normalization
        center = points.mean(axis=0)
        points = points - center
        scale = np.max(np.abs(points))
        if scale > 0:
            points = points / scale
        
        # Step 4-5: Compute normals, curvature, and roughness
        normals, curvature, roughness = self._estimate_features_numpy(points)
        
        # Combine features
        features = np.concatenate([
            points,
            normals,
            curvature.reshape(-1, 1),
            roughness.reshape(-1, 1)
        ], axis=1)
        
        return {
            'points': points.astype(np.float32),
            'normals': normals.astype(np.float32),
            'curvature': curvature.astype(np.float32),
            'roughness': roughness.astype(np.float32),
            'features': features.astype(np.float32)
        }
    
    def _estimate_features_numpy(
        self, 
        points: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Estimate normals, curvature, and roughness using PCA."""
        from scipy.spatial import cKDTree
        
        n_points = len(points)
        tree = cKDTree(points)
        
        normals = np.zeros((n_points, 3), dtype=np.float32)
        curvature = np.zeros(n_points, dtype=np.float32)
        roughness = np.zeros(n_points, dtype=np.float32)
        
        _, indices = tree.query(points, k=self.normal_knn)
        
        for i in range(n_points):
            idx = indices[i]
            neighbors = points[idx]
            
            # PCA for normal estimation
            centered = neighbors - neighbors.mean(axis=0)
            cov = np.cov(centered.T)
            
            eigenvalues, eigenvectors = np.linalg.eigh(cov)
            sort_idx = np.argsort(eigenvalues)
            eigenvalues = eigenvalues[sort_idx]
            eigenvectors = eigenvectors[:, sort_idx]
            
            # Normal is eigenvector with smallest eigenvalue
            normals[i] = eigenvectors[:, 0]
            
            # Curvature proxy
            total = eigenvalues.sum()
            if total > 1e-10:
                curvature[i] = eigenvalues[0] / total
        
        # Orient normals consistently (pointing outward)
        # Use simple heuristic: normals should point away from centroid
        centroid = points.mean(axis=0)
        for i in range(n_points):
            direction = points[i] - centroid
            if np.dot(normals[i], direction) < 0:
                normals[i] = -normals[i]
        
        # Compute roughness as normal variance in neighborhood
        for i in range(n_points):
            idx = indices[i]
            neighbor_normals = normals[idx]
            roughness[i] = np.var(neighbor_normals, axis=0).sum()
        
        # Normalize roughness
        if roughness.max() > 0:
            roughness = roughness / roughness.max()
        
        return normals, curvature, roughness
    
    def to_tensor(self, data: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
        """Convert numpy arrays to PyTorch tensors."""
        return {k: torch.from_numpy(v) for k, v in data.items()}


def preprocess_batch(
    point_clouds: list,
    preprocessor: PointCloudPreprocessor,
    num_workers: int = 4
) -> list:
    """
    Preprocess a batch of point clouds in parallel.
    
    Args:
        point_clouds: List of point clouds (N_i, 3)
        preprocessor: PointCloudPreprocessor instance
        num_workers: Number of parallel workers
        
    Returns:
        List of preprocessed data dictionaries
    """
    from concurrent.futures import ThreadPoolExecutor
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(preprocessor.process, point_clouds))
    
    return results
