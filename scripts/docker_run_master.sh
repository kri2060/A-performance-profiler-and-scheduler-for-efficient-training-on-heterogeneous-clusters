#!/bin/bash

# Docker Multi-Node Training - MASTER Node Script
# Run this on your MAIN laptop (the one coordinating training)

set -e

echo "=========================================="
echo "Starting MASTER Node for Distributed Training"
echo "=========================================="

# Configuration
MASTER_PORT=${MASTER_PORT:-29500}
WORLD_SIZE=${WORLD_SIZE:-4}  # Total number of nodes (including this one)
EXPERIMENT_NAME=${EXPERIMENT_NAME:-"distributed_training"}
IMAGE_NAME=${IMAGE_NAME:-"hetero-cluster-training"}

# Get the host's IP address
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    MASTER_ADDR=$(hostname -I | awk '{print $1}')
elif [[ "$OSTYPE" == "darwin"* ]]; then
    MASTER_ADDR=$(ipconfig getifaddr en0)
else
    echo "Please manually set MASTER_ADDR environment variable"
    exit 1
fi

echo "Master IP Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "World Size: $WORLD_SIZE"
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

# Run master container
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
    $IMAGE_NAME \
    bash -c "
        echo 'Master node ready. Starting training...'
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

echo "Master node completed!"
