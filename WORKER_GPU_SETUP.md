# Worker GPU Detection Troubleshooting Guide

## Problem
Worker has GPU visible in `nvidia-smi` but PyTorch shows:
```
WARNING:profiling.performance_profiler:Failed to initialize NVML: Invalid Argument
torch.AcceleratorError: CUDA error: invalid device ordinal
```

## Diagnostic Steps

### Step 1: Copy and run the diagnostic script on the worker
```bash
# On worker machine
cd ~/hetero-training
source venv/bin/activate
python diagnose_cuda.py
```

This will show you exactly what's wrong.

## Common Issues & Solutions

### Issue 1: PyTorch CPU-Only Version
**Symptom**: `diagnose_cuda.py` shows "Built with CUDA: NO"

**Solution**: Reinstall PyTorch with CUDA support
```bash
# Uninstall current PyTorch
pip uninstall torch torchvision torchaudio

# Check your CUDA version with nvidia-smi (top right corner)
nvidia-smi  # Look for "CUDA Version: X.X"

# Install PyTorch with matching CUDA (choose one):
# For CUDA 12.1:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation:
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Issue 2: CUDA Libraries Not Found
**Symptom**: `diagnose_cuda.py` shows "Built with CUDA: YES" but "CUDA available: False"

**Solution**: The updated `START_WORKER.sh` script now includes CUDA path detection. But you may need to manually set it:

```bash
# Find your CUDA installation
ls /usr/local/ | grep cuda

# Add to START_WORKER.sh (if not auto-detected):
export CUDA_HOME=/usr/local/cuda-XX.X  # Replace XX.X with your version
export LD_LIBRARY_PATH=/usr/local/cuda-XX.X/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-XX.X/bin:$PATH
```

### Issue 3: WSL2 GPU Access (Windows)
**Symptom**: Running on WSL2 and GPU not accessible

**Solution**: Ensure WSL2 CUDA support is enabled
```bash
# Check if CUDA is accessible in WSL2
nvidia-smi

# If not working, you may need to:
# 1. Update Windows to latest version
# 2. Install NVIDIA drivers on Windows (not in WSL2)
# 3. Install CUDA toolkit in WSL2
```

### Issue 4: CUDA Version Mismatch
**Symptom**: PyTorch built with CUDA X.X but system has CUDA Y.Y

**Solution**: Either:
- Reinstall PyTorch matching your system CUDA version (preferred)
- Install the CUDA version PyTorch expects

## Updated Configuration

The following files have been updated:

### 1. `START_WORKER.sh`
- ✅ Auto-detects CUDA installation paths
- ✅ Sets up NCCL for GPU training
- ✅ Configures proper network interface

### 2. `master.sh`
- ✅ Uses NCCL backend (both nodes have GPUs)
- ✅ Correct IP address: 10.161.199.69

### 3. `distributed_trainer.py`
- ✅ Falls back to CPU if CUDA unavailable
- ✅ Handles mixed CPU/GPU setups

## Testing the Setup

### On Worker Machine:
```bash
# 1. Run diagnostics
python diagnose_cuda.py

# 2. If CUDA available, run simple GPU test
python test_gpu.py

# 3. Start worker node
./START_WORKER.sh
```

### On Master Machine:
```bash
# Run diagnostics (optional)
python diagnose_cuda.py

# Start master node
./master.sh
```

## Expected Output

When working correctly, you should see:
```
INFO:training.distributed_trainer:Rank 1: Using CUDA device 0
[NCCL INFO] Bootstrap: Using eth0:XXX.XXX.XXX.XXX
[NCCL INFO] NET/Socket : Using [0]eth0:XXX.XXX.XXX.XXX
```

Not:
```
WARNING:profiling.performance_profiler:Failed to initialize NVML: Invalid Argument
torch.AcceleratorError: CUDA error: invalid device ordinal
```

## Still Not Working?

1. Share output of `diagnose_cuda.py` from worker
2. Check if different CUDA path is needed
3. Verify PyTorch CUDA version matches system CUDA
4. Ensure firewall port 29500 is open on master

## Quick Fix Script for Worker

If PyTorch was installed without CUDA, run this on the worker:

```bash
#!/bin/bash
cd ~/hetero-training
source venv/bin/activate

# Reinstall PyTorch with CUDA 12.1
pip uninstall -y torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Verify
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```
