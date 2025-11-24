#!/bin/bash
# Fix Docker GPU Support
# This script installs NVIDIA Container Toolkit to enable GPU access in Docker

set -e

echo "=========================================="
echo "🔧 Fixing Docker GPU Support"
echo "=========================================="
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script needs sudo privileges to install packages"
    echo "Please run with sudo or as root:"
    echo "  sudo ./fix_docker_gpu.sh"
    exit 1
fi

# Check if NVIDIA GPU is present
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found"
    echo "Please install NVIDIA drivers first"
    exit 1
fi

echo "✅ NVIDIA GPU detected:"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
echo ""

# Detect Linux distribution
if [ -f /etc/os-release ]; then
    . /etc/os-release
    DISTRO=$ID
    VERSION_ID=$VERSION_ID
else
    echo "❌ Could not detect Linux distribution"
    exit 1
fi

echo "📦 Installing NVIDIA Container Toolkit for $DISTRO..."
echo ""

# Install based on distribution
case "$DISTRO" in
    ubuntu|debian)
        echo "Installing for Ubuntu/Debian..."

        # Add NVIDIA GPG key
        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
          && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

        # Update and install
        sudo apt-get update
        sudo apt-get install -y nvidia-container-toolkit
        ;;

    arch|manjaro)
        echo "Installing for Arch Linux..."

        # Install from AUR or official repos
        if command -v yay &> /dev/null; then
            yay -S --noconfirm nvidia-container-toolkit
        elif command -v paru &> /dev/null; then
            paru -S --noconfirm nvidia-container-toolkit
        else
            # Manual installation for Arch
            pacman -S --noconfirm nvidia-container-toolkit || \
            echo "Please install nvidia-container-toolkit manually:"
            echo "  yay -S nvidia-container-toolkit"
        fi
        ;;

    fedora|rhel|centos)
        echo "Installing for Fedora/RHEL/CentOS..."

        distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
        curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
          sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

        sudo yum install -y nvidia-container-toolkit
        ;;

    *)
        echo "❌ Unsupported distribution: $DISTRO"
        echo "Please install nvidia-container-toolkit manually"
        echo "See: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
        ;;
esac

echo ""
echo "✅ NVIDIA Container Toolkit installed"
echo ""

# Configure Docker
echo "⚙️  Configuring Docker to use NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
echo ""

# Restart Docker
echo "🔄 Restarting Docker daemon..."
sudo systemctl restart docker
echo ""

# Wait for Docker to start
echo "⏳ Waiting for Docker to start..."
sleep 5

# Test GPU access in Docker
echo "🧪 Testing GPU access in Docker..."
echo ""

if docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi; then
    echo ""
    echo "=========================================="
    echo "✅ SUCCESS! Docker can now access GPU"
    echo "=========================================="
    echo ""
    echo "You can now run GPU-accelerated containers!"
    echo ""
    echo "Next steps:"
    echo "  1. Try running the master setup again:"
    echo "     ./scripts/docker_run_master_adaptive.sh"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "❌ GPU test failed"
    echo "=========================================="
    echo ""
    echo "Troubleshooting:"
    echo "  1. Reboot your system: sudo reboot"
    echo "  2. Check Docker service: sudo systemctl status docker"
    echo "  3. Check logs: sudo journalctl -u docker"
    echo ""
fi
