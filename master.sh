#!/bin/bash

echo "=========================================="
echo "NCCL Master Node - Rank 0"
echo "=========================================="

source venv/bin/activate

# Kill previous runs
pkill -9 -f "python.*src.training.main" 2>/dev/null
sleep 2

# ============================================
# CRITICAL: Network Configuration
# ============================================
export GLOO_SOCKET_IFNAME=wlan0           # Network interface for Gloo backend
export TP_SOCKET_IFNAME=wlan0             # Tensor parallel interface

# Optional: NCCL config (if needed for GPU operations)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=wlan0
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Distributed training config
export RANK=0                             # Master is always rank 0
export WORLD_SIZE=2                      # ← CHANGE THIS: Total number of nodes (master + workers)

# Auto-detect master IP from wlan0 interface
MASTER_IP=$(ip addr show wlan0 | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | tail -n1)
export MASTER_ADDR=${MASTER_IP}          # Auto-detected master IP
export MASTER_PORT=29500                  # Port for communication
export LOCAL_RANK=0                       # GPU index on this node

echo "Master IP: $MASTER_ADDR"
echo "Waiting for $((WORLD_SIZE - 1)) workers to connect..."
echo "=========================================="

# Start training with lightweight synthetic dataset for demo
python -m src.training.main \
  --model simple_cnn \
  --dataset synthetic_image \
  --num-samples 1000 \
  --image-size 32 \
  --batch-size 32 \
  --epochs 5 \
  --lr 0.01 \
  --backend gloo \
  --master-addr $MASTER_ADDR \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --experiment-name demo_gloo_heterogeneous