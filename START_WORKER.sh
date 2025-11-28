#!/bin/bash
echo "=========================================="
echo "Starting Worker Node (Rank 1) - Linux"
echo "=========================================="

# Activate virtual environment
source venv/bin/activate

# ============================================
# CUDA Configuration (Linux)
# ============================================
# Check for CUDA installation and set paths
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
    echo "Found CUDA at: /usr/local/cuda"
elif [ -d "/usr/local/cuda-12" ]; then
    CUDA_VERSION=$(ls -d /usr/local/cuda-12* | sort -V | tail -n1)
    export CUDA_HOME=$CUDA_VERSION
    export LD_LIBRARY_PATH=$CUDA_VERSION/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_VERSION/bin:$PATH
    echo "Found CUDA at: $CUDA_VERSION"
elif [ -d "/usr/local/cuda-11" ]; then
    CUDA_VERSION=$(ls -d /usr/local/cuda-11* | sort -V | tail -n1)
    export CUDA_HOME=$CUDA_VERSION
    export LD_LIBRARY_PATH=$CUDA_VERSION/lib64:$LD_LIBRARY_PATH
    export PATH=$CUDA_VERSION/bin:$PATH
    echo "Found CUDA at: $CUDA_VERSION"
else
    echo "WARNING: CUDA not found. Running in CPU-only mode."
fi

# Ensure CUDA devices are visible (if available)
if command -v nvidia-smi &> /dev/null; then
    export CUDA_VISIBLE_DEVICES=0
    echo "CUDA device available:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
else
    echo "No NVIDIA GPU detected. Using CPU."
fi

# ============================================
# Network Configuration (Linux)
# ============================================
# Auto-detect network interface (prefer wireless, fallback to ethernet)
NETWORK_INTERFACE=""
if ip link show wlan0 &>/dev/null; then
    NETWORK_INTERFACE="wlan0"
elif ip link show wlp &>/dev/null; then
    # Some Linux systems use wlp* naming
    NETWORK_INTERFACE=$(ip link show | grep -o "wlp[^:]*" | head -n1)
elif ip link show wlx &>/dev/null; then
    # USB WiFi adapters use wlx* naming
    NETWORK_INTERFACE=$(ip link show | grep -o "wlx[^:]*" | head -n1)
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

# NCCL Configuration for GPU training
export NCCL_DEBUG=INFO                    # Show detailed NCCL logs
export NCCL_SOCKET_IFNAME=$NETWORK_INTERFACE
export NCCL_IB_DISABLE=1                  # Disable InfiniBand (use Ethernet)
export NCCL_P2P_DISABLE=1                 # Disable peer-to-peer GPU transfer

# Gloo backend configuration
export GLOO_SOCKET_IFNAME=$NETWORK_INTERFACE
export TP_SOCKET_IFNAME=$NETWORK_INTERFACE
export GLOO_DEVICE_TRANSPORT=TCP

# ============================================
# Distributed Training Configuration
# ============================================
export RANK=1                             # Worker rank (change for additional workers)
export WORLD_SIZE=2                       # Total nodes (must match master)
export MASTER_ADDR=192.168.1.100         # ← CHANGE THIS: Set to master node's IP
export MASTER_PORT=29500
export LOCAL_RANK=0                       # Local device index

echo "Network Interface: $NETWORK_INTERFACE"
echo "Connecting to master at: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="

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
