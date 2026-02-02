#!/bin/bash
echo "=========================================="
echo "Worker CUDA Setup Script"
echo "Matching Master Configuration"
echo "=========================================="

# Check if GPU exists
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ ERROR: nvidia-smi not found!"
    echo "Please install NVIDIA drivers first"
    exit 1
fi

echo "✓ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Check current PyTorch installation
echo ""
echo "Checking current PyTorch installation..."
if python -c "import torch" 2>/dev/null; then
    CURRENT_VERSION=$(python -c "import torch; print(torch.__version__)")
    CUDA_AVAILABLE=$(python -c "import torch; print(torch.cuda.is_available())")
    echo "Current PyTorch: $CURRENT_VERSION"
    echo "CUDA available: $CUDA_AVAILABLE"

    if [[ "$CURRENT_VERSION" == *"+cu118"* ]] && [[ "$CUDA_AVAILABLE" == "True" ]]; then
        echo "✓ PyTorch with CUDA 11.8 already installed: $CURRENT_VERSION"
        echo "No installation needed."
        exit 0
    else
        echo "⚠ Different PyTorch version detected. Reinstalling..."
    fi
else
    echo "PyTorch not installed. Installing..."
fi

# Uninstall existing PyTorch if present
echo ""
echo "Uninstalling existing PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null

# Install PyTorch with CUDA 11.8 support
# Note: PyTorch 2.x is compatible across minor versions for distributed training
echo ""
echo "Installing PyTorch with CUDA 11.8..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation..."
echo "=========================================="

python -c "
import torch
import sys

print(f'PyTorch Version: {torch.__version__}')
print(f'CUDA Available: {torch.cuda.is_available()}')

if torch.cuda.is_available():
    print(f'CUDA Version: {torch.version.cuda}')
    print(f'cuDNN Version: {torch.backends.cudnn.version()}')
    print(f'GPU Device Count: {torch.cuda.device_count()}')
    print(f'GPU Device Name: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')

    # Test CUDA
    print('')
    print('Testing CUDA tensor operations...')
    x = torch.rand(5, 3).cuda()
    print(f'✓ CUDA tensor created successfully: {x.device}')
    print('')
    print('✅ CUDA is working correctly!')
else:
    print('')
    print('❌ ERROR: CUDA not available!')
    print('Please check:')
    print('  1. NVIDIA drivers are installed (nvidia-smi)')
    print('  2. CUDA toolkit is compatible')
    sys.exit(1)
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    INSTALLED_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "Worker is now configured:"
    echo "  - PyTorch: $INSTALLED_VERSION"
    echo "  - CUDA: 11.8"
    echo ""
    echo "Next steps:"
    echo "  1. Fix network connectivity (ping master)"
    echo "  2. Update START_WORKER.sh with master IP"
    echo "  3. Run ./START_WORKER.sh"
else
    echo ""
    echo "❌ Installation failed. Please check the errors above."
    exit 1
fi
