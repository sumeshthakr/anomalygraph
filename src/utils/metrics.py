"""
Evaluation Metrics for Anomaly Detection

Implements standard metrics for 3D anomaly detection:
1. AUROC (Area Under ROC Curve)
2. PRO (Per-Region Overlap)
3. FPR at fixed TPR
"""

import numpy as np
from typing import Optional, Tuple, List
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from scipy.ndimage import label as connected_components


def compute_auroc(
    labels: np.ndarray,
    scores: np.ndarray
) -> float:
    """
    Compute Area Under the Receiver Operating Characteristic curve.
    
    Args:
        labels: Ground truth labels (0 = normal, 1 = anomaly)
        scores: Predicted anomaly scores
        
    Returns:
        AUROC value
    """
    if len(np.unique(labels)) < 2:
        return 0.5  # Cannot compute AUROC with single class
    
    return roc_auc_score(labels, scores)


def compute_fpr_at_tpr(
    labels: np.ndarray,
    scores: np.ndarray,
    tpr: float = 0.95
) -> float:
    """
    Compute False Positive Rate at a fixed True Positive Rate.
    
    This metric is particularly relevant for industrial applications
    where high sensitivity is required.
    
    Args:
        labels: Ground truth labels
        scores: Predicted anomaly scores
        tpr: Target true positive rate (default: 0.95)
        
    Returns:
        FPR at the specified TPR
    """
    if len(np.unique(labels)) < 2:
        return 1.0
    
    fpr, tpr_values, _ = roc_curve(labels, scores)
    
    # Find FPR at target TPR
    idx = np.searchsorted(tpr_values, tpr)
    if idx >= len(fpr):
        idx = len(fpr) - 1
    
    return fpr[idx]


def compute_pro(
    labels: np.ndarray,
    scores: np.ndarray,
    num_thresholds: int = 100
) -> float:
    """
    Compute Per-Region Overlap score.
    
    PRO measures localization quality by computing the overlap
    between predicted and ground truth anomaly regions at various
    thresholds.
    
    Args:
        labels: Ground truth per-point labels (0 = normal, 1 = anomaly)
        scores: Per-point anomaly scores
        num_thresholds: Number of thresholds to evaluate
        
    Returns:
        PRO score (0-1, higher is better)
    """
    if labels.sum() == 0:
        return 1.0  # No anomalies, perfect score
    
    # Generate thresholds
    thresholds = np.linspace(scores.min(), scores.max(), num_thresholds)
    
    pro_values = []
    
    for threshold in thresholds:
        predictions = (scores > threshold).astype(int)
        
        # Find connected components in ground truth
        gt_regions, n_regions = _get_connected_regions(labels)
        
        if n_regions == 0:
            continue
        
        # Compute overlap for each region
        overlaps = []
        for region_id in range(1, n_regions + 1):
            region_mask = gt_regions == region_id
            region_size = region_mask.sum()
            
            if region_size == 0:
                continue
            
            overlap = (predictions[region_mask] == 1).sum() / region_size
            overlaps.append(overlap)
        
        if overlaps:
            pro_values.append(np.mean(overlaps))
    
    if not pro_values:
        return 0.0
    
    # Return average PRO across thresholds
    return np.mean(pro_values)


def _get_connected_regions(labels: np.ndarray) -> Tuple[np.ndarray, int]:
    """
    Find connected regions in labels.
    
    For point clouds, this uses simple distance-based clustering.
    """
    # For 1D labels, each contiguous segment is a region
    # In practice, for 3D point clouds, you'd use spatial clustering
    
    if labels.sum() == 0:
        return np.zeros_like(labels), 0
    
    # Simple approach: label contiguous segments
    regions = np.zeros_like(labels)
    current_region = 0
    in_region = False
    
    for i, label in enumerate(labels):
        if label == 1:
            if not in_region:
                current_region += 1
                in_region = True
            regions[i] = current_region
        else:
            in_region = False
    
    return regions, current_region


def compute_ap(
    labels: np.ndarray,
    scores: np.ndarray
) -> float:
    """
    Compute Average Precision.
    
    Args:
        labels: Ground truth labels
        scores: Predicted anomaly scores
        
    Returns:
        Average Precision value
    """
    if len(np.unique(labels)) < 2:
        return 0.0
    
    precision, recall, _ = precision_recall_curve(labels, scores)
    
    # Compute AP using trapezoidal rule
    ap = np.trapz(precision, recall)
    
    return ap


def compute_f1_optimal(
    labels: np.ndarray,
    scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute optimal F1 score and corresponding threshold.
    
    Args:
        labels: Ground truth labels
        scores: Predicted anomaly scores
        
    Returns:
        Tuple of (best F1 score, optimal threshold)
    """
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    
    # Compute F1 for each threshold
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else thresholds[-1]
    
    return best_f1, best_threshold


def compute_all_metrics(
    object_labels: np.ndarray,
    object_scores: np.ndarray,
    point_labels: Optional[np.ndarray] = None,
    point_scores: Optional[np.ndarray] = None
) -> dict:
    """
    Compute all evaluation metrics.
    
    Args:
        object_labels: Object-level ground truth
        object_scores: Object-level anomaly scores
        point_labels: Per-point ground truth (optional)
        point_scores: Per-point anomaly scores (optional)
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {}
    
    # Object-level metrics
    if len(np.unique(object_labels)) > 1:
        metrics['object_auroc'] = compute_auroc(object_labels, object_scores)
        metrics['object_fpr_95'] = compute_fpr_at_tpr(object_labels, object_scores, 0.95)
        metrics['object_ap'] = compute_ap(object_labels, object_scores)
        f1, threshold = compute_f1_optimal(object_labels, object_scores)
        metrics['object_f1'] = f1
        metrics['object_threshold'] = threshold
    
    # Point-level metrics
    if point_labels is not None and point_scores is not None:
        if len(np.unique(point_labels)) > 1:
            metrics['point_auroc'] = compute_auroc(point_labels, point_scores)
            metrics['point_fpr_95'] = compute_fpr_at_tpr(point_labels, point_scores, 0.95)
            metrics['pro'] = compute_pro(point_labels, point_scores)
            metrics['point_ap'] = compute_ap(point_labels, point_scores)
    
    return metrics


class CategoryMetrics:
    """
    Track and aggregate metrics per category for cross-category evaluation.
    """
    
    def __init__(self):
        self.categories = {}
    
    def add(
        self,
        category: str,
        object_label: int,
        object_score: float,
        point_labels: Optional[np.ndarray] = None,
        point_scores: Optional[np.ndarray] = None
    ):
        """Add a sample's results."""
        if category not in self.categories:
            self.categories[category] = {
                'object_labels': [],
                'object_scores': [],
                'point_labels': [],
                'point_scores': []
            }
        
        self.categories[category]['object_labels'].append(object_label)
        self.categories[category]['object_scores'].append(object_score)
        
        if point_labels is not None:
            self.categories[category]['point_labels'].append(point_labels)
        if point_scores is not None:
            self.categories[category]['point_scores'].append(point_scores)
    
    def compute(self) -> dict:
        """Compute metrics for each category and overall."""
        results = {'per_category': {}, 'overall': {}}
        
        all_object_labels = []
        all_object_scores = []
        
        for category, data in self.categories.items():
            object_labels = np.array(data['object_labels'])
            object_scores = np.array(data['object_scores'])
            
            all_object_labels.extend(object_labels)
            all_object_scores.extend(object_scores)
            
            # Per-category metrics
            point_labels = None
            point_scores = None
            if data['point_labels']:
                point_labels = np.concatenate(data['point_labels'])
                point_scores = np.concatenate(data['point_scores'])
            
            results['per_category'][category] = compute_all_metrics(
                object_labels, object_scores,
                point_labels, point_scores
            )
        
        # Overall metrics
        results['overall'] = compute_all_metrics(
            np.array(all_object_labels),
            np.array(all_object_scores)
        )
        
        # Mean per-category AUROC
        aurocs = [
            m.get('object_auroc', 0) 
            for m in results['per_category'].values()
        ]
        results['overall']['mean_category_auroc'] = np.mean(aurocs) if aurocs else 0
        
        return results
    
    def summary(self) -> str:
        """Get summary string of metrics."""
        results = self.compute()
        
        lines = ["=" * 50]
        lines.append("Evaluation Results")
        lines.append("=" * 50)
        
        # Per-category
        lines.append("\nPer-Category Results:")
        lines.append("-" * 50)
        for category, metrics in results['per_category'].items():
            auroc = metrics.get('object_auroc', 0)
            lines.append(f"  {category}: AUROC = {auroc:.4f}")
        
        # Overall
        lines.append("\nOverall Results:")
        lines.append("-" * 50)
        for key, value in results['overall'].items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.4f}")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
