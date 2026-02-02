#!/bin/bash
echo "=========================================="
echo "Worker GPU Diagnostic Script"
echo "=========================================="

# Check NVIDIA driver
echo ""
echo "1. Checking NVIDIA Driver..."
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found"
    nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader
    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)
    echo "Driver Version: $DRIVER_VERSION"
else
    echo "❌ nvidia-smi not found!"
    echo "Install NVIDIA drivers first"
    exit 1
fi

# Check CUDA compatibility
echo ""
echo "2. Checking CUDA Compatibility..."
echo "For CUDA 11.8, you need driver >= 450.80.02"
echo "For CUDA 12.x, you need driver >= 525.60"

# Check if CUDA toolkit is installed
echo ""
echo "3. Checking for CUDA Toolkit..."
if [ -d "/usr/local/cuda" ]; then
    echo "✓ CUDA found at /usr/local/cuda"
    if [ -f "/usr/local/cuda/version.txt" ]; then
        cat /usr/local/cuda/version.txt
    elif command -v nvcc &> /dev/null; then
        nvcc --version | grep "release"
    fi
else
    echo "⚠ CUDA toolkit not found in /usr/local/cuda"
    echo "This is OK - PyTorch includes its own CUDA runtime"
fi

# Check Python and PyTorch
echo ""
echo "4. Checking Python Environment..."
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Python: $(python --version)"

    if python -c "import torch" 2>/dev/null; then
        echo ""
        python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA compiled: {torch.version.cuda}')
print(f'CUDA available: {torch.cuda.is_available()}')

if not torch.cuda.is_available():
    print('')
    print('⚠ CUDA not available. Possible reasons:')
    print('  1. Driver too old for CUDA 11.8')
    print('  2. GPU not detected')
    print('  3. CUDA runtime mismatch')

    # Check if we can see the GPU from PyTorch
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(f'  GPU detected by nvidia-smi: {result.stdout.strip()}')
    except:
        pass
"
    else
        echo "❌ PyTorch not installed"
    fi
else
    echo "❌ Virtual environment not found"
fi

# Recommendations
echo ""
echo "=========================================="
echo "Recommendations:"
echo "=========================================="
echo ""
echo "If CUDA is not available, try these fixes:"
echo ""
echo "Option 1: Update NVIDIA Driver (Recommended)"
echo "  sudo apt update"
echo "  sudo apt install nvidia-driver-535  # or latest"
echo "  sudo reboot"
echo ""
echo "Option 2: Use CPU-only PyTorch (For CPU workers)"
echo "  pip uninstall torch torchvision torchaudio"
echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu"
echo "  # Worker will use Gloo backend (CPU)"
echo ""
echo "Option 3: Install matching CUDA toolkit"
echo "  # If driver supports CUDA 11.8"
echo "  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin"
echo "  sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600"
echo "  wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb"
echo "  sudo dpkg -i cuda-repo-ubuntu2204-11-8-local_11.8.0-520.61.05-1_amd64.deb"
echo "  sudo cp /var/cuda-repo-ubuntu2204-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/"
echo "  sudo apt-get update"
echo "  sudo apt-get -y install cuda"
echo ""
