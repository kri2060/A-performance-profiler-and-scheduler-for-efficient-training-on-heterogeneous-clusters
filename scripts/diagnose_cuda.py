#!/usr/bin/env python3
"""
Comprehensive CUDA Diagnostic Script
Helps identify why PyTorch isn't detecting GPU
"""
import sys
import os
import subprocess

print("=" * 70)
print("CUDA DIAGNOSTIC TOOL")
print("=" * 70)

# 1. Check nvidia-smi
print("\n1. NVIDIA-SMI Check:")
print("-" * 70)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ nvidia-smi is available")
        print(result.stdout)
    else:
        print("✗ nvidia-smi failed")
        print(result.stderr)
except FileNotFoundError:
    print("✗ nvidia-smi not found in PATH")

# 2. Check CUDA environment variables
print("\n2. CUDA Environment Variables:")
print("-" * 70)
cuda_vars = ['CUDA_HOME', 'CUDA_PATH', 'CUDA_VISIBLE_DEVICES',
             'LD_LIBRARY_PATH', 'PATH']
for var in cuda_vars:
    value = os.environ.get(var, 'NOT SET')
    print(f"{var}: {value}")

# 3. Check for CUDA libraries
print("\n3. CUDA Library Search:")
print("-" * 70)
common_cuda_paths = [
    '/usr/local/cuda',
    '/usr/local/cuda-12.0',
    '/usr/local/cuda-11.0',
    '/opt/cuda',
    'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA',
]
for path in common_cuda_paths:
    if os.path.exists(path):
        print(f"✓ Found: {path}")
        # Check for libcudart
        lib_paths = [
            os.path.join(path, 'lib64', 'libcudart.so'),
            os.path.join(path, 'lib', 'x64', 'cudart64_*.dll'),
        ]
        for lib in lib_paths:
            if os.path.exists(lib) or len(subprocess.run(
                ['find', path, '-name', 'libcudart*'],
                capture_output=True
            ).stdout) > 0:
                print(f"  ✓ CUDA runtime found in {path}")
                break

# 4. Check PyTorch installation
print("\n4. PyTorch Installation:")
print("-" * 70)
try:
    import torch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"  Built with CUDA: {torch.version.cuda if torch.version.cuda else 'NO'}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"  CUDA device count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"    Capability: {torch.cuda.get_device_capability(i)}")
    else:
        print("\n  ✗ CUDA NOT AVAILABLE IN PYTORCH")
        print("\n  Possible reasons:")
        print("  - PyTorch installed without CUDA support (CPU-only version)")
        print("  - CUDA libraries not in LD_LIBRARY_PATH")
        print("  - CUDA version mismatch with PyTorch")
        print("  - Missing NVIDIA drivers")

except ImportError:
    print("✗ PyTorch not installed")

# 5. Check if PyTorch was built with CUDA
print("\n5. PyTorch Build Info:")
print("-" * 70)
try:
    import torch
    print(f"Debug: {torch.version.debug}")
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
except:
    print("Could not get PyTorch build info")

# 6. Recommendations
print("\n6. Recommendations:")
print("-" * 70)
try:
    import torch
    if not torch.cuda.is_available():
        if torch.version.cuda:
            print("PyTorch was built with CUDA but can't find it.")
            print("\nTry adding to your script:")
            print('  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH')
            print('  export CUDA_HOME=/usr/local/cuda')
        else:
            print("PyTorch was built WITHOUT CUDA (CPU-only version).")
            print("\nReinstall PyTorch with CUDA:")
            print('  pip uninstall torch torchvision')
            print('  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')
    else:
        print("✓ Everything looks good!")
except:
    pass

print("\n" + "=" * 70)
