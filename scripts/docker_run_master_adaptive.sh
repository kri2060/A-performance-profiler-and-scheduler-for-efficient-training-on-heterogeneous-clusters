#!/bin/bash

# Docker Multi-Node Training - ADAPTIVE MASTER Node Script
# Automatically detects GPU availability and uses GPU or CPU accordingly

set -e

echo "=========================================="
echo "Starting ADAPTIVE MASTER Node"
echo "=========================================="

# Configuration
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-4}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"distributed_training"}
IMAGE_NAME=${IMAGE_NAME:-"hetero-cluster-training"}

# Get the host's IP address with better compatibility
if command -v ip &> /dev/null; then
    # Try using ip command (most modern Linux)
    MASTER_ADDR=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v '127.0.0.1' | head -n1)
elif command -v hostname &> /dev/null; then
    # Try hostname with different flags
    if hostname -I &> /dev/null; then
        MASTER_ADDR=$(hostname -I 2>/dev/null | awk '{print $1}')
    elif hostname -i &> /dev/null; then
        MASTER_ADDR=$(hostname -i 2>/dev/null | awk '{print $1}' | grep -v '127.0.0.1' | head -n1)
    fi
fi

# Fallback if automatic detection fails
if [ -z "$MASTER_ADDR" ] || [ "$MASTER_ADDR" == "127.0.0.1" ]; then
    echo "Warning: Could not auto-detect IP address"
    echo "Please manually enter your IP address (e.g., 192.168.1.100):"
    read -r MASTER_ADDR
fi

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

echo "Master IP Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
echo "Device: $([ "$GPU_AVAILABLE" = true ] && echo "GPU" || echo "CPU")"
echo ""
echo "IMPORTANT: Share this information with worker nodes:"
echo "  MASTER_ADDR=$MASTER_ADDR"
echo "  MASTER_PORT=$MASTER_PORT"
echo "  WORLD_SIZE=$WORLD_SIZE"
echo "=========================================="

# Build Docker image if it doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Create shared volume for experiments
mkdir -p experiments/logs experiments/configs

# Run based on GPU availability
if [ "$GPU_AVAILABLE" = true ]; then
    echo "Starting master in GPU mode..."
    docker run --rm -it \
        --gpus all \
        --network host \
        --name hetero-master \
        -v "$(pwd):/workspace" \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e MASTER_PORT=$MASTER_PORT \
        -e RANK=0 \
        -e WORLD_SIZE=$WORLD_SIZE \
        -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
        -e CUDA_VISIBLE_DEVICES=0 \
        $IMAGE_NAME \
        bash -c "
            echo 'Master node ready (GPU). Starting training...'
            echo 'Waiting 10 seconds for workers to connect...'
            sleep 10

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
    echo "Starting master in CPU mode..."
    docker run --rm -it \
        --network host \
        --name hetero-master \
        -v "$(pwd):/workspace" \
        -e MASTER_ADDR=$MASTER_ADDR \
        -e MASTER_PORT=$MASTER_PORT \
        -e RANK=0 \
        -e WORLD_SIZE=$WORLD_SIZE \
        -e EXPERIMENT_NAME=$EXPERIMENT_NAME \
        -e CUDA_VISIBLE_DEVICES="" \
        $IMAGE_NAME \
        bash -c "
            echo 'Master node ready (CPU). Starting training...'
            echo 'Waiting 10 seconds for workers to connect...'
            sleep 10

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

echo "Master node completed!"
