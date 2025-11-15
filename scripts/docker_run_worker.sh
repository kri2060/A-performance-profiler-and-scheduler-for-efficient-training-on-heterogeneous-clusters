#!/bin/bash

# Docker Multi-Node Training - WORKER Node Script
# Run this on each WORKER laptop (Windows/Linux)

set -e

echo "=========================================="
echo "Starting WORKER Node for Distributed Training"
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

echo "Master Address: $MASTER_ADDR"
echo "Master Port: $MASTER_PORT"
echo "Worker Rank: $RANK"
echo "World Size: $WORLD_SIZE"
echo "=========================================="

# Build Docker image if it doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Create shared volume for experiments
mkdir -p experiments/logs experiments/configs

# Run worker container
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
    $IMAGE_NAME \
    bash -c "
        echo 'Worker node $RANK ready. Connecting to master at $MASTER_ADDR:$MASTER_PORT...'

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

echo "Worker node $RANK completed!"
