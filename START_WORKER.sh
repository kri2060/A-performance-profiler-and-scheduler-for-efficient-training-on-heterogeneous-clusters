#!/bin/bash
# Worker start script for Windows WSL2
# Run this on the Windows worker machine

echo "=========================================="
echo "Starting Worker Node (Rank 1)"
echo "=========================================="

source venv/bin/activate

# ============================================
# CUDA Configuration (WSL2 specific)
# ============================================
# WSL2 CUDA libraries are in /usr/lib/wsl/lib
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# Also check standard CUDA paths
if [ -d "/usr/local/cuda" ]; then
    export CUDA_HOME=/usr/local/cuda
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda/bin:$PATH
elif [ -d "/usr/local/cuda-12.0" ]; then
    export CUDA_HOME=/usr/local/cuda-12.0
    export LD_LIBRARY_PATH=/usr/local/cuda-12.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-12.0/bin:$PATH
elif [ -d "/usr/local/cuda-11.0" ]; then
    export CUDA_HOME=/usr/local/cuda-11.0
    export LD_LIBRARY_PATH=/usr/local/cuda-11.0/lib64:$LD_LIBRARY_PATH
    export PATH=/usr/local/cuda-11.0/bin:$PATH
fi

# WSL2 specific: Ensure CUDA is visible
export CUDA_VISIBLE_DEVICES=0

# NCCL Configuration for GPU training
export NCCL_DEBUG=INFO                    # Show detailed NCCL logs
export NCCL_SOCKET_IFNAME=eth0            # Network interface (change if needed)
export NCCL_IB_DISABLE=1                  # Disable InfiniBand (use Ethernet)
export NCCL_P2P_DISABLE=1                 # Disable peer-to-peer GPU transfer

# Distributed training config
export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.161.199.69  # Linux master IP
export MASTER_PORT=29500
export LOCAL_RANK=0

# WSL2 needs to advertise the Windows host IP for incoming connections
# This tells Gloo to use the Windows host IP instead of WSL2 internal IP
export GLOO_SOCKET_IFNAME=eth0
export TP_SOCKET_IFNAME=eth0
export GLOO_DEVICE_TRANSPORT=TCP

echo "Connecting to master at: $MASTER_ADDR:$MASTER_PORT"
echo "=========================================="

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
