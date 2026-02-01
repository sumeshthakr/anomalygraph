# AnomalyGraph: Graph-Based 3D Point-Cloud Anomaly Detection

A generalizable graph-based system for detecting and localizing geometric anomalies in 3D point clouds. Trained on ShapeNet-style anomaly data, designed to generalize to unseen object categories.

![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)
![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)
![License MIT](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Features

- **Class-agnostic detection**: Works on arbitrary object categories
- **Multi-scale analysis**: Detects both micro-defects (scratches, pits) and macro-defects (bulges, dents, missing parts)
- **Self-supervised pretraining**: Learns geometry representations without anomaly labels
- **Few-shot calibration**: Adapts to new object types with only 10-30 normal samples
- **Per-point heatmaps**: Precise localization of anomalous regions

## 🏗️ Architecture

```
Input Point Cloud (N×3)
        │
        ▼
┌─────────────────────────────────────┐
│     Preprocessing Pipeline          │
│  • Outlier removal                  │
│  • Voxel downsampling               │
│  • Scale normalization              │
│  • Feature extraction (normals,     │
│    curvature, roughness)            │
└─────────────────────────────────────┘
        │
        ▼ [x,y,z,nx,ny,nz,curv,rough]
┌─────────────────────────────────────┐
│     Multi-Scale Graph Construction  │
│  • Fine Graph (k=24, all points)    │
│  • Coarse Graph (k=16, 2048 FPS)    │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Hierarchical Graph Encoder      │
│  • EdgeConv / Graph Attention       │
│  • Fine → Pool → Coarse → Global    │
│  • Embedding dim: 256               │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Anomaly Detection               │
│  • Memory bank distance (local)     │
│  • Mahalanobis distance (global)    │
│  • Score fusion (0.7L + 0.3G)       │
└─────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────┐
│     Output                          │
│  • Per-point anomaly heatmap        │
│  • Object-level anomaly score       │
└─────────────────────────────────────┘
```

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/anomalygraph.git
cd anomalygraph

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install PyTorch Geometric (adjust CUDA version as needed)
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric
```

## 📊 Dataset

This project uses the [Anomaly-ShapeNet](https://github.com/Chopper-233/Anomaly-ShapeNet) dataset.

```bash
# Download the dataset
git clone https://github.com/Chopper-233/Anomaly-ShapeNet.git data/Anomaly-ShapeNet
```

### Expected Dataset Structure
```
data/Anomaly-ShapeNet/
├── category1/
│   ├── train/
│   │   └── good/
│   │       └── *.npy
│   └── test/
│       ├── good/
│       │   └── *.npy
│       └── anomaly_type/
│           └── *.npy
├── category2/
│   └── ...
└── ...
```

## 🚀 Quick Start

### Training with Jupyter Notebook

```bash
cd notebooks
jupyter notebook train_anomaly_detection.ipynb
```

### Training on RunPod

```bash
# Option 1: Using the training script directly
python scripts/train_runpod.py \
    --data_root ./data/Anomaly-ShapeNet \
    --epochs 100 \
    --batch_size 4 \
    --output_dir ./outputs

# Option 2: Using the shell script
chmod +x scripts/run_training.sh
./scripts/run_training.sh
```

### Inference

```bash
python scripts/inference.py \
    --model checkpoints/final_model.pth \
    --input sample.ply \
    --output results/ \
    --visualize
```

## 📁 Project Structure

```
anomalygraph/
├── configs/
│   └── default.yaml          # Default configuration
├── notebooks/
│   └── train_anomaly_detection.ipynb  # Training notebook
├── scripts/
│   ├── train_runpod.py       # RunPod training script
│   ├── run_training.sh       # Shell script for training
│   └── inference.py          # Inference script
├── src/
│   ├── data/
│   │   ├── preprocessing.py  # Point cloud preprocessing
│   │   ├── graph_construction.py  # Graph building
│   │   └── dataset.py        # Dataset classes
│   ├── models/
│   │   ├── edge_conv.py      # EdgeConv layers
│   │   ├── graph_attention.py  # Graph attention layers
│   │   ├── encoder.py        # Hierarchical encoder
│   │   └── anomaly_detector.py  # Full detector
│   ├── training/
│   │   ├── pretraining.py    # Self-supervised objectives
│   │   ├── trainer.py        # Training pipeline
│   │   └── calibration.py    # Category calibration
│   └── utils/
│       ├── metrics.py        # Evaluation metrics
│       ├── visualization.py  # Visualization tools
│       └── memory_bank.py    # Memory bank implementation
└── requirements.txt
```

## ⚙️ Configuration

Edit `configs/default.yaml` to customize:

```yaml
# Key parameters
data:
  target_points: 100000        # Points after downsampling
  normal_knn: 30               # Neighbors for normal estimation

graph:
  fine_k: 24                   # Fine graph neighbors
  coarse_k: 16                 # Coarse graph neighbors
  num_superpoints: 2048        # Superpoints for coarse graph

model:
  encoder_type: "edgeconv"     # or "attention"
  out_channels: 256            # Embedding dimension

anomaly:
  memory_bank_size: 10000      # Number of prototypes
  local_weight: 0.7            # Weight for local scores
  global_weight: 0.3           # Weight for global score
```

## 📈 Expected Performance

| Metric | Target | Description |
|--------|--------|-------------|
| Object AUROC | >0.70 | Object-level detection |
| Point AUROC | >0.65 | Per-point localization |
| PRO | >0.60 | Per-region overlap |

### Defect Detection Capabilities

| Defect Type | Feasibility | Notes |
|-------------|-------------|-------|
| Broken parts | High | Strong global deviation |
| Missing chunks | High | Topology discontinuity |
| Bulges/dents | High | Multi-scale curvature |
| Medium holes | Medium-High | Sampling density dependent |
| Scratches | Medium | Requires high-quality normals |

## 🔧 Training Pipeline

### Stage 1: Self-Supervised Pretraining

```python
# Masked Patch Modeling
- Mask ratio: 0.4
- Patch size: 32 points
- Predict: coordinates + normals

# Contrastive Learning
- Augmentations: jitter, dropout, scaling, rotation
- Temperature: 0.07
```

### Stage 2: Normal Feature Modeling

```python
# Memory Bank
- K-means clustering (10k prototypes)
- Anomaly score = distance to nearest prototypes

# Global Gaussian
- Fit to global embeddings
- Mahalanobis distance scoring
```

### Stage 3: Calibration for New Categories

```python
# Per-category calibration
- Collect 10-30 normal samples
- Update memory bank
- Compute threshold (99.5 percentile)
```

## 📊 Evaluation Metrics

```python
from src.utils import compute_auroc, compute_pro

# Object-level AUROC
auroc = compute_auroc(labels, scores)

# Per-region Overlap (PRO)
pro = compute_pro(point_labels, point_scores)

# FPR at 95% TPR
fpr = compute_fpr_at_tpr(labels, scores, tpr=0.95)
```

## 🎨 Visualization

```python
from src.utils import visualize_heatmap

visualize_heatmap(
    points,
    heatmap,
    title="Anomaly Detection Result",
    colormap="jet",
    save_path="output.png"
)
```

## ⚠️ Known Limitations

- Point spacing > defect size may miss fine scratches
- Heavy sensor noise can corrupt normal estimation
- Severe occlusions may create boundary artifacts
- Requires per-category calibration for best results

## 📝 Citation

If you use this code, please cite:

```bibtex
@software{anomalygraph2024,
  title={AnomalyGraph: Graph-Based 3D Point-Cloud Anomaly Detection},
  year={2024},
  url={https://github.com/yourusername/anomalygraph}
}
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📧 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.