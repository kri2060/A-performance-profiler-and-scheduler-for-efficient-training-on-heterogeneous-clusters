#!/bin/bash

# Docker Multi-Node Training - Dashboard Script
# Run this to monitor training progress via Streamlit dashboard

set -e

echo "=========================================="
echo "Starting Monitoring Dashboard"
echo "=========================================="

IMAGE_NAME=${IMAGE_NAME:-"hetero-cluster-training"}
DASHBOARD_PORT=${DASHBOARD_PORT:-8501}

# Build Docker image if it doesn't exist
if ! docker image inspect $IMAGE_NAME > /dev/null 2>&1; then
    echo "Building Docker image..."
    docker build -t $IMAGE_NAME .
fi

# Create shared volume for experiments
mkdir -p experiments/logs experiments/configs

echo "Starting Streamlit dashboard on port $DASHBOARD_PORT..."
echo "Access the dashboard at: http://localhost:$DASHBOARD_PORT"
echo "=========================================="

# Run dashboard container
docker run --rm -it \
    --name hetero-dashboard \
    -p $DASHBOARD_PORT:8501 \
    -v "$(pwd):/workspace" \
    $IMAGE_NAME \
    streamlit run src/monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0

echo "Dashboard stopped!"
