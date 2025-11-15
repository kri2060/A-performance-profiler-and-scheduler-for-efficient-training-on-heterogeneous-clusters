#!/bin/bash

# Setup script for heterogeneous cluster
# Automates cluster configuration and node discovery

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "========================================="
echo "Heterogeneous Cluster Setup"
echo "========================================="

# Configuration
CONFIG_DIR="$PROJECT_ROOT/experiments/configs"
mkdir -p "$CONFIG_DIR"

# Function to check Python installation
check_python() {
    if ! command -v python3 &> /dev/null; then
        echo "Error: Python 3 not found. Please install Python 3.9+"
        exit 1
    fi

    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo "✓ Python $PYTHON_VERSION found"
}

# Function to check CUDA installation
check_cuda() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✓ NVIDIA drivers found"
        nvidia-smi --query-gpu=name --format=csv,noheader | nl
    else
        echo "⚠ NVIDIA drivers not found. GPU support will be limited."
    fi
}

# Function to check PyTorch installation
check_pytorch() {
    python3 -c "import torch; print(f'✓ PyTorch {torch.__version__} installed')" 2>/dev/null || {
        echo "⚠ PyTorch not found. Installing..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
    }

    python3 -c "import torch; print(f'✓ CUDA available: {torch.cuda.is_available()}')" 2>/dev/null
}

# Function to install dependencies
install_dependencies() {
    echo ""
    echo "Installing dependencies..."
    echo "-------------------------"

    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
        echo "✓ Dependencies installed"
    else
        echo "⚠ requirements.txt not found"
    fi

    # Install project
    if [ -f "$PROJECT_ROOT/setup.py" ]; then
        pip install -e "$PROJECT_ROOT" --quiet
        echo "✓ Project installed"
    fi
}

# Function to profile node
profile_node() {
    echo ""
    echo "Profiling current node..."
    echo "-------------------------"

    python3 -m src.profiling.main --output-dir "$CONFIG_DIR" || {
        echo "⚠ Profiling failed. Continuing anyway..."
    }
}

# Function to test training
test_training() {
    echo ""
    echo "Testing training setup..."
    echo "-------------------------"

    python3 -m src.training.main \
        --model simple_cnn \
        --dataset synthetic_image \
        --num-samples 100 \
        --batch-size 8 \
        --epochs 1 \
        --experiment-name setup_test \
        --backend gloo 2>&1 | head -20

    echo "✓ Training test completed"
}

# Function to create startup script
create_startup_script() {
    echo ""
    echo "Creating startup scripts..."
    echo "---------------------------"

    # Worker startup script
    cat > "$PROJECT_ROOT/start_worker.sh" << 'EOF'
#!/bin/bash
# Worker node startup script

# Configuration
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-"29500"}
RANK=${RANK:-0}
WORLD_SIZE=${WORLD_SIZE:-1}

echo "Starting worker node..."
echo "  Rank: $RANK"
echo "  Master: $MASTER_ADDR:$MASTER_PORT"
echo "  World Size: $WORLD_SIZE"

# Set environment variables
export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=$MASTER_PORT
export RANK=$RANK
export WORLD_SIZE=$WORLD_SIZE
export LOCAL_RANK=$RANK

# Wait for master
echo "Waiting for master node..."
timeout 60 bash -c "until nc -z $MASTER_ADDR $MASTER_PORT; do sleep 1; done" || {
    echo "Error: Could not connect to master"
    exit 1
}

echo "✓ Connected to master"
echo "Ready for training!"
EOF

    chmod +x "$PROJECT_ROOT/start_worker.sh"

    echo "✓ Created start_worker.sh"
}

# Function to print usage instructions
print_usage() {
    echo ""
    echo "========================================="
    echo "Setup Complete!"
    echo "========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Profile your cluster:"
    echo "   python -m src.profiling.main"
    echo ""
    echo "2. Run quick test:"
    echo "   python -m src.training.main --model simple_cnn --dataset synthetic_image --epochs 2"
    echo ""
    echo "3. Launch dashboard:"
    echo "   streamlit run src/monitoring/dashboard.py"
    echo ""
    echo "4. Run benchmarks:"
    echo "   bash scripts/run_benchmark.sh"
    echo ""
    echo "Multi-node setup:"
    echo "-----------------"
    echo "On MASTER node:"
    echo "  export MASTER_ADDR=<master-ip>"
    echo "  export MASTER_PORT=29500"
    echo "  export RANK=0"
    echo "  export WORLD_SIZE=<total-nodes>"
    echo "  python -m src.training.main [options]"
    echo ""
    echo "On WORKER nodes:"
    echo "  export MASTER_ADDR=<master-ip>"
    echo "  export MASTER_PORT=29500"
    echo "  export RANK=<node-rank>"
    echo "  export WORLD_SIZE=<total-nodes>"
    echo "  python -m src.training.main [options]"
    echo ""
    echo "Or use the startup script:"
    echo "  MASTER_ADDR=<ip> RANK=<rank> WORLD_SIZE=<size> bash start_worker.sh"
    echo ""
    echo "For help:"
    echo "  python -m src.training.main --help"
    echo ""
}

# Main setup flow
main() {
    echo ""
    echo "Step 1: Checking Python..."
    check_python

    echo ""
    echo "Step 2: Checking CUDA..."
    check_cuda

    echo ""
    echo "Step 3: Checking PyTorch..."
    check_pytorch

    echo ""
    echo "Step 4: Installing dependencies..."
    install_dependencies

    echo ""
    echo "Step 5: Profiling hardware..."
    profile_node

    echo ""
    echo "Step 6: Testing training..."
    test_training

    echo ""
    echo "Step 7: Creating startup scripts..."
    create_startup_script

    print_usage
}

# Run main
main
