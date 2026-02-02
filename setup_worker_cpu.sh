#!/bin/bash
echo "=========================================="
echo "Worker CPU-Only Setup Script"
echo "For workers without GPU or CUDA issues"
echo "=========================================="

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
    echo "Current PyTorch: $CURRENT_VERSION"

    # Check if it's CPU-only
    if [[ "$CURRENT_VERSION" == *"+cpu"* ]]; then
        echo "✓ CPU-only PyTorch already installed!"
        echo "No installation needed."
        exit 0
    else
        echo "⚠ Different PyTorch version detected. Reinstalling CPU version..."
    fi
else
    echo "PyTorch not installed. Installing..."
fi

# Uninstall existing PyTorch if present
echo ""
echo "Uninstalling existing PyTorch packages..."
pip uninstall -y torch torchvision torchaudio 2>/dev/null

# Install CPU-only PyTorch
echo ""
echo "Installing CPU-only PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
echo ""
echo "Installing other dependencies..."
pip install -r requirements.txt

# Verify installation
echo ""
echo "=========================================="
echo "Verifying Installation..."
echo "=========================================="

python -c "
import torch
import sys

print(f'PyTorch Version: {torch.__version__}')
print(f'CPU Mode: Yes')
print(f'Number of CPU threads: {torch.get_num_threads()}')

# Test CPU tensor operations
print('')
print('Testing CPU tensor operations...')
x = torch.rand(5, 3)
y = torch.rand(5, 3)
z = x + y
print(f'✓ CPU tensor operations working')
print(f'  Tensor device: {x.device}')
print('')
print('✅ CPU-only PyTorch is working correctly!')
"

if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Setup Complete!"
    echo "=========================================="
    INSTALLED_VERSION=$(python -c "import torch; print(torch.__version__)")
    echo "Worker is now configured:"
    echo "  - PyTorch: $INSTALLED_VERSION"
    echo "  - Mode: CPU-only"
    echo "  - Backend: Gloo (CPU distributed training)"
    echo ""
    echo "⚠ Note: This worker will use CPU for training."
    echo "   It will automatically use Gloo backend instead of NCCL."
    echo ""
    echo "Next steps:"
    echo "  1. Fix network connectivity (ping master)"
    echo "  2. Update START_WORKER.sh with master IP"
    echo "  3. Run ./START_WORKER.sh"
    echo ""
    echo "The worker will automatically fall back to CPU/Gloo backend."
else
    echo ""
    echo "❌ Installation failed. Please check the errors above."
    exit 1
fi
