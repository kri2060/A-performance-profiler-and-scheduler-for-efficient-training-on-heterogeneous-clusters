#!/bin/bash
echo "=========================================="
echo "Master Node - Rank 0 (Linux)"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# Kill previous runs
pkill -9 -f "python.*src.training.main" 2>/dev/null
sleep 2

# ============================================
# CRITICAL: Network Configuration
# ============================================
# Auto-detect network interface (prefer wireless, fallback to ethernet)
NETWORK_INTERFACE=""
if ip link show wlan0 &>/dev/null; then
    NETWORK_INTERFACE="wlan0"
elif ip link show wlp &>/dev/null; then
    # Some Linux systems use wlp* naming
    NETWORK_INTERFACE=$(ip link show | grep -o "wlp[^:]*" | head -n1)
elif ip link show eth0 &>/dev/null; then
    NETWORK_INTERFACE="eth0"
elif ip link show enp &>/dev/null; then
    # Modern Linux ethernet naming
    NETWORK_INTERFACE=$(ip link show | grep -o "enp[^:]*" | head -n1)
else
    echo "ERROR: Could not detect network interface!"
    echo "Available interfaces:"
    ip link show
    exit 1
fi

echo "Using network interface: $NETWORK_INTERFACE"

export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE    # Network interface for Gloo backend
export TP_SOCKET_IFNAME=$NETWORK_INTERFACE      # Tensor parallel interface

# Optional: NCCL config (if needed for GPU operations)
export NCCL_DEBUG=INFO
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1

# Distributed training config
export RANK=0                             # Master is always rank 0
export WORLD_SIZE=2                      # ← CHANGE THIS: Total number of nodes (master + workers)

# Auto-detect master IP from the selected interface
MASTER_IP=$(ip addr show $NETWORK_INTERFACE | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | head -n1)

if [ -z "$MASTER_IP" ]; then
    echo "ERROR: Could not detect IP address on interface $NETWORK_INTERFACE"
    echo "Interface status:"
    ip addr show $NETWORK_INTERFACE
    exit 1
fi

export MASTER_ADDR=${MASTER_IP}          # Auto-detected master IP
export MASTER_PORT=29500                  # Port for communication
export LOCAL_RANK=0                       # GPU/Device index on this node

echo "Network Interface: $NETWORK_INTERFACE"
echo "Master IP: $MASTER_ADDR"
echo "Waiting for $((WORLD_SIZE - 1)) workers to connect..."
echo "=========================================="

# Start training with lightweight synthetic dataset for demo
# Use -u flag for unbuffered output to see logs immediately
# Note: backend defaults to 'nccl' in main.py
python -u -m src.training.main \
  --model simple_cnn \
  --dataset synthetic_image \
  --num-samples 1000 \
  --image-size 32 \
  --batch-size 32 \
  --epochs 5 \
  --lr 0.01 \
  --master-addr $MASTER_ADDR \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --experiment-name demo_nccl_heterogeneous