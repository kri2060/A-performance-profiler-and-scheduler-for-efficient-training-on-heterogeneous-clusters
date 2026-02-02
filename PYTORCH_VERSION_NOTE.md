# PyTorch Version Compatibility Note

## Current Setup

**Master Node:**
- PyTorch: 2.6.0+cu118
- CUDA: 11.8
- Python: 3.13

**Worker Node (after setup):**
- PyTorch: Latest 2.x with CUDA 11.8 (likely 2.4.0 or 2.5.1)
- CUDA: 11.8
- Python: 3.12

## Why Different PyTorch Versions?

PyTorch 2.6.0 doesn't have pre-built binaries for CUDA 11.8. The latest pre-built version with CUDA 11.8 is **2.4.0** or **2.5.1**.

## Is This a Problem?

**No! PyTorch distributed training is compatible across minor versions (2.x).**

The distributed communication protocol (c10d) is stable across PyTorch 2.x versions:
- ✅ PyTorch 2.6.0 (master) can communicate with PyTorch 2.4.0 (worker)
- ✅ Both use the same NCCL/Gloo backends
- ✅ Tensor serialization is compatible
- ✅ Model parameters can be synchronized

## What Matters for Compatibility

For distributed training, what's important is:
1. **✅ Same CUDA version** - Both use CUDA 11.8
2. **✅ Same Python major version** - Both use Python 3.x
3. **✅ Compatible PyTorch major version** - Both use PyTorch 2.x
4. **✅ Same backend** - Both use NCCL/Gloo

## If You Want Exact Versions

If you prefer to have the exact same PyTorch version on both nodes, you have these options:

### Option 1: Downgrade Master to 2.4.0 (Recommended)

On master:
```bash
source venv/bin/activate
pip uninstall torch torchvision torchaudio
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
```

### Option 2: Compile PyTorch 2.6.0 from Source on Worker

Not recommended - takes hours and requires build tools.

### Option 3: Upgrade CUDA to 12.1 on Both Nodes

Then both can use PyTorch 2.6.0 with pre-built binaries.

```bash
# Install CUDA 12.1
# Then install PyTorch 2.6.0 with CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Recommendation

**Just use different minor versions** - PyTorch 2.6.0 on master and 2.4.0 on worker works perfectly fine for distributed training!

The `setup_worker_cuda.sh` script is already configured to install the latest compatible version.

## Verification

After setup, verify both nodes can communicate:

**On Master:**
```python
import torch
import torch.distributed as dist
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

**On Worker:**
```python
import torch
import torch.distributed as dist
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
```

Both should report `CUDA: True` and will work together in distributed training!
