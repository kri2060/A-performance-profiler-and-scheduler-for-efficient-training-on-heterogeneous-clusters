#!/bin/bash

# Docker Multi-Node Training - ADAPTIVE WORKER Node Script
# Automatically detects GPU availability and uses GPU or CPU accordingly
# Run this on each WORKER laptop (Windows/Linux)

set -e

echo "=========================================="
echo "Starting ADAPTIVE WORKER Node"
echo "=========================================="

# Configuration - MUST BE SET BY USER
if [ -z "$MASTER_ADDR" ]; then
    echo "ERROR: MASTER_ADDR not set!"
    echo "Please set MASTER_ADDR to the master node's IP address:"
    echo "  export MASTER_ADDR=192.168.1.100"
    exit 1
fi

if [ -z "$RANK" ]; then
    echo "ERROR: RANK not set!"
    echo "Please set RANK to this worker's rank (1, 2, 3, ...):"
    echo "  export RANK=1"
    exit 1
fi

MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"distributed_training"}
IMAGE_NAME=${IMAGE_NAME:-"hetero-cluster-training"}

# Detect GPU availability
GPU_AVAILABLE=false
if command -v nvidia-smi &> /dev/null; then
    if nvidia-smi &> /dev/null; then
        GPU_AVAILABLE=true
        echo "✓ GPU detected - will use GPU acceleration"
    else
        echo "✗ nvidia-smi found but GPU not accessible - will use CPU"
    fi
else
    echo "✗ No GPU detected - will use CPU"
fi

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Worker Rank: $RANK"
echo "World Size: $WORLD_SIZE"
echo "Device: $([ "$GPU_AVAILABLE" = true ] && echo "GPU" || echo "CPU")"
echo "=========================================="

# Build Docker image if it doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Create shared volume for experiments
mkdir -p experiments/logs experiments/configs

# Prepare docker run command based on GPU availability
if [ "$GPU_AVAILABLE" = true ]; then
    # GPU mode
    echo "Starting worker in GPU mode..."
    docker run --rm -it \
        --gpus all \
        --network host \
        --name hetero-worker-$RANK \
        -v "$(pwd):/workspace" \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e MASTER_PORT=$MASTER_PORT \
        -e RANK=$RANK \
        -e WORLD_SIZE=$WORLD_SIZE \
        -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
        -e CUDA_VISIBLE_DEVICES=0 \
        $IMAGE_NAME \
        bash -c "
            echo 'Worker node $RANK ready (GPU). Connecting to master at $MASTER_ADDR:$MASTER_PORT...'

            python -m src.training.main \
                --model simple_cnn \
                --dataset synthetic_image \
                --batch-size 64 \
                --epochs 10 \
                --enable-profiling \
                --enable-load-balancing \
                --load-balance-policy dynamic \
                --backend gloo \
                --master-addr \$MASTER_ADDR \
                --master-port \$MASTER_PORT \
                --experiment-name \$EXPERIMENT_NAME
        "
else
    # CPU mode
    echo "Starting worker in CPU mode..."
    docker run --rm -it \
        --network host \
        --name hetero-worker-$RANK \
        -v "$(pwd):/workspace" \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e MASTER_PORT=$MASTER_PORT \
        -e RANK=$RANK \
        -e WORLD_SIZE=$WORLD_SIZE \
        -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
        -e CUDA_VISIBLE_DEVICES="" \
        $IMAGE_NAME \
        bash -c "
            echo 'Worker node $RANK ready (CPU). Connecting to master at $MASTER_ADDR:$MASTER_PORT...'

            python -m src.training.main \
                --model simple_cnn \
                --dataset synthetic_image \
                --batch-size 32 \
                --epochs 10 \
                --enable-profiling \
                --enable-load-balancing \
                --load-balance-policy dynamic \
                --backend gloo \
                --master-addr \$MASTER_ADDR \
                --master-port \$MASTER_PORT \
                --experiment-name \$EXPERIMENT_NAME
        "
fi

echo "Worker node $RANK completed!"
