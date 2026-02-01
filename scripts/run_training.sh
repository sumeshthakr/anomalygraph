#!/bin/bash
# RunPod setup and training script
# This script sets up the environment and runs training on RunPod

set -e

echo "========================================"
echo "Graph-Based 3D Point-Cloud Anomaly Detection"
echo "RunPod Training Setup"
echo "========================================"

# Configuration
DATA_DIR="${DATA_DIR:-./data}"
OUTPUT_DIR="${OUTPUT_DIR:-./outputs}"
EPOCHS="${EPOCHS:-100}"
BATCH_SIZE="${BATCH_SIZE:-4}"
LR="${LR:-0.0001}"

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "WARNING: No GPU detected. Training will be slow on CPU."
fi

# Install dependencies
echo ""
echo "Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Install PyTorch Geometric dependencies
echo ""
echo "Installing PyTorch Geometric..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torch-geometric

# Download dataset if not present
echo ""
echo "Checking for Anomaly-ShapeNet dataset..."
if [ ! -d "${DATA_DIR}/Anomaly-ShapeNet" ]; then
    echo "Downloading Anomaly-ShapeNet dataset..."
    mkdir -p "${DATA_DIR}"
    cd "${DATA_DIR}"
    git clone https://github.com/Chopper-233/Anomaly-ShapeNet.git
    cd -
else
    echo "Dataset found at ${DATA_DIR}/Anomaly-ShapeNet"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Run training
echo ""
echo "========================================"
echo "Starting Training"
echo "========================================"
echo "Epochs: ${EPOCHS}"
echo "Batch Size: ${BATCH_SIZE}"
echo "Learning Rate: ${LR}"
echo "Output Dir: ${OUTPUT_DIR}"
echo "========================================"

python scripts/train_runpod.py \
    --data_root "${DATA_DIR}/Anomaly-ShapeNet" \
    --config configs/default.yaml \
    --epochs ${EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --lr ${LR} \
    --output_dir "${OUTPUT_DIR}" \
    --num_workers 4

echo ""
echo "========================================"
echo "Training Complete!"
echo "========================================"
echo "Results saved to: ${OUTPUT_DIR}"
