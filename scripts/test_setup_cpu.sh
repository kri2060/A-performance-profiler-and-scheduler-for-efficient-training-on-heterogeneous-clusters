#!/bin/bash
# Quick test of multi-node setup using CPU mode (no GPU required)

set -e

echo "=========================================="
echo "🧪 Testing Multi-Node Setup (CPU Mode)"
echo "=========================================="
echo ""
echo "This script will test your setup without requiring GPU support in Docker"
echo ""

# Configuration
export WORLD_SIZE=2
export MASTER_PORT=29500
export EXPERIMENT_NAME="test_cpu_training"

# Get IP
MASTER_ADDR=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n1)

echo "Configuration:"
echo "  World Size: $WORLD_SIZE"
echo "  Master IP: $MASTER_ADDR"
echo "  Master Port: $MASTER_PORT"
echo ""

# Check if Docker is running
if ! docker ps &> /dev/null; then
    echo "❌ Error: Docker is not running"
    echo "Please start Docker and try again"
    exit 1
fi

echo "✅ Docker is running"
echo ""

# Build Docker image (CPU version)
echo "📦 Building Docker image (CPU mode)..."
docker build -f Dockerfile.cpu -t hetero-cluster-training-cpu .

echo ""
echo "=========================================="
echo "🚀 Starting Master Node (CPU Mode)"
echo "=========================================="
echo ""
echo "Share with workers:"
echo "  export MASTER_ADDR=$MASTER_ADDR"
echo "  export MASTER_PORT=$MASTER_PORT"
echo "  export WORLD_SIZE=$WORLD_SIZE"
echo ""
echo "Waiting 10 seconds for you to start workers..."
echo "=========================================="
echo ""

# Run master in CPU mode
docker run --rm \
  --name hetero-master-cpu \
  --network host \
  -v "$(pwd):/workspace" \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e RANK=0 \
  -e WORLD_SIZE=$WORLD_SIZE \
  -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
  hetero-cluster-training-cpu \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --num-samples 1000 \
    --batch-size 32 \
    --epochs 2 \
    --backend gloo \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    --experiment-name $EXPERIMENT_NAME
