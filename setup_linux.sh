#!/bin/bash
#
# Linux Setup Script for Heterogeneous Cluster Training
#

set -e  # Exit on error

echo "=========================================="
echo "Linux Setup - Heterogeneous Cluster Training"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo -e "${RED}ERROR: This script is for Linux only${NC}"
    exit 1
fi

# Check Python version
echo -e "\n${YELLOW}Checking Python version...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}Found Python $PYTHON_VERSION${NC}"
else
    echo -e "${RED}ERROR: Python3 not found. Please install Python 3.9+${NC}"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}Virtual environment created${NC}"
else
    echo -e "${GREEN}Virtual environment already exists${NC}"
fi

# Activate virtual environment
echo -e "\n${YELLOW}Activating virtual environment...${NC}"
source venv/bin/activate

# Upgrade pip
echo -e "\n${YELLOW}Upgrading pip...${NC}"
pip install --upgrade pip > /dev/null 2>&1
echo -e "${GREEN}pip upgraded${NC}"

# Detect GPU
echo -e "\n${YELLOW}Checking for NVIDIA GPU...${NC}"
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}NVIDIA GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    GPU_AVAILABLE=true

    # Check CUDA
    if command -v nvcc &> /dev/null; then
        CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)
        echo -e "${GREEN}CUDA $CUDA_VERSION detected${NC}"
    else
        echo -e "${YELLOW}WARNING: nvcc not found. CUDA toolkit may not be installed${NC}"
    fi
else
    echo -e "${YELLOW}No NVIDIA GPU detected. Will install CPU-only version${NC}"
    GPU_AVAILABLE=false
fi

# Install PyTorch
echo -e "\n${YELLOW}Installing PyTorch...${NC}"
if [ "$GPU_AVAILABLE" = true ]; then
    echo -e "${YELLOW}Installing PyTorch with CUDA support...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
else
    echo -e "${YELLOW}Installing PyTorch (CPU-only)...${NC}"
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi
echo -e "${GREEN}PyTorch installed${NC}"

# Install dependencies
echo -e "\n${YELLOW}Installing project dependencies...${NC}"
if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
    echo -e "${GREEN}Dependencies installed${NC}"
else
    echo -e "${YELLOW}WARNING: requirements.txt not found${NC}"
fi

# Make scripts executable
echo -e "\n${YELLOW}Making scripts executable...${NC}"
chmod +x master.sh START_WORKER.sh 2>/dev/null || true
chmod +x scripts/*.sh 2>/dev/null || true
echo -e "${GREEN}Scripts are now executable${NC}"

# Detect network interface
echo -e "\n${YELLOW}Detecting network interface...${NC}"
if ip link show wlan0 &>/dev/null; then
    NETWORK_INTERFACE="wlan0"
elif ip link show wlp &>/dev/null; then
    NETWORK_INTERFACE=$(ip link show | grep -o "wlp[^:]*" | head -n1)
elif ip link show eth0 &>/dev/null; then
    NETWORK_INTERFACE="eth0"
elif ip link show enp &>/dev/null; then
    NETWORK_INTERFACE=$(ip link show | grep -o "enp[^:]*" | head -n1)
else
    NETWORK_INTERFACE="unknown"
fi

if [ "$NETWORK_INTERFACE" != "unknown" ]; then
    IP_ADDRESS=$(ip addr show $NETWORK_INTERFACE | grep "inet " | grep -v "127.0.0.1" | awk '{print $2}' | cut -d/ -f1 | head -n1)
    echo -e "${GREEN}Network Interface: $NETWORK_INTERFACE${NC}"
    echo -e "${GREEN}IP Address: $IP_ADDRESS${NC}"
else
    echo -e "${YELLOW}WARNING: Could not auto-detect network interface${NC}"
    echo "Available interfaces:"
    ip link show | grep -E "^[0-9]+:" | cut -d: -f2 | tr -d ' '
fi

# Create necessary directories
echo -e "\n${YELLOW}Creating project directories...${NC}"
mkdir -p experiments/configs
mkdir -p experiments/logs
mkdir -p experiments/metrics
mkdir -p data
echo -e "${GREEN}Directories created${NC}"

# Firewall check
echo -e "\n${YELLOW}Checking firewall status...${NC}"
if command -v ufw &> /dev/null; then
    UFW_STATUS=$(sudo ufw status | grep -i "status:" | awk '{print $2}')
    if [ "$UFW_STATUS" = "active" ]; then
        echo -e "${YELLOW}Firewall is active. You may need to allow port 29500:${NC}"
        echo "  sudo ufw allow 29500/tcp"
        echo "  sudo ufw allow from <worker-ip>"
    else
        echo -e "${GREEN}Firewall is inactive${NC}"
    fi
else
    echo -e "${YELLOW}ufw not found. Check your firewall settings manually${NC}"
fi

# Test PyTorch installation
echo -e "\n${YELLOW}Testing PyTorch installation...${NC}"
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
EOF
echo -e "${GREEN}PyTorch test complete${NC}"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "Next steps:"
echo ""
echo "1. To run as MASTER node:"
echo "   - Edit master.sh and set WORLD_SIZE"
echo "   - Run: ./master.sh"
echo ""
echo "2. To run as WORKER node:"
echo "   - Edit START_WORKER.sh:"
echo "     * Set MASTER_ADDR to master's IP"
echo "     * Set RANK (1, 2, 3, etc.)"
echo "     * Set WORLD_SIZE to match master"
echo "   - Run: ./START_WORKER.sh"
echo ""
echo "3. For detailed instructions, see LINUX_SETUP.md"
echo ""
if [ "$NETWORK_INTERFACE" != "unknown" ]; then
    echo "Your current IP address: $IP_ADDRESS"
    echo "Share this with worker nodes as MASTER_ADDR"
fi
echo ""
echo "=========================================="
