# Linux Setup Guide - Heterogeneous Cluster Training

This guide will help you set up the heterogeneous cluster training system on Linux (Ubuntu/Debian).

## Prerequisites

- Ubuntu 20.04+ or similar Linux distribution
- Python 3.9+
- Git
- Network connectivity between master and worker nodes

## System Requirements

### Master Node
- Python 3.9+
- 4GB+ RAM
- Network interface (WiFi or Ethernet)

### Worker Node(s)
- Python 3.9+
- 4GB+ RAM
- CUDA-capable GPU (optional, will use CPU if not available)
- Network interface (WiFi or Ethernet)

## Installation Steps

### 1. Clone the Repository

On both master and worker nodes:

```bash
cd ~
git clone <repository-url> hetero-cluster
cd hetero-cluster
```

### 2. Set Up Python Virtual Environment

On both master and worker nodes:

```bash
# Install Python virtual environment if not already installed
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip
```

### 3. Install Dependencies

On both nodes:

```bash
# Install PyTorch (CPU version - works on all systems)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support (if you have NVIDIA GPU):
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt
```

### 4. Install CUDA (Optional - for GPU support on worker)

If you have an NVIDIA GPU and want GPU acceleration:

```bash
# Check if NVIDIA drivers are installed
nvidia-smi

# If not installed, install NVIDIA drivers
sudo apt-get install -y nvidia-driver-535  # or latest version

# Install CUDA toolkit (example for CUDA 11.8)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-toolkit-11-8

# Verify CUDA installation
nvcc --version
nvidia-smi
```

### 5. Configure Firewall

Allow communication between nodes:

```bash
# On both master and worker nodes
sudo ufw allow 29500/tcp  # PyTorch distributed port
sudo ufw allow from <other-node-ip>  # Replace with actual IP

# Or disable firewall temporarily for testing
sudo ufw disable
```

### 6. Find Network Interface and IP Address

On both nodes, identify your network interface:

```bash
# List all network interfaces
ip link show

# Common interfaces:
# - wlan0: WiFi
# - eth0: Ethernet
# - enp*: Modern ethernet naming
# - wlp*: Modern wireless naming

# Get your IP address
ip addr show wlan0  # or your interface name
# Look for "inet" line, e.g., "inet 192.168.1.100/24"
```

## Running the Cluster

### On Master Node

1. Edit [master.sh](master.sh) if needed:
   - Line 48: Set `WORLD_SIZE` to total number of nodes (master + workers)

2. Make the script executable and run:

```bash
chmod +x master.sh
./master.sh
```

The master will:
- Auto-detect your network interface
- Display its IP address
- Wait for workers to connect

Example output:
```
==========================================
Master Node - Rank 0 (Linux)
==========================================
Using network interface: wlan0
Network Interface: wlan0
Master IP: 192.168.1.100
Waiting for 1 workers to connect...
==========================================
```

### On Worker Node(s)

1. Note the master IP address from the master node output

2. Edit [START_WORKER.sh](START_WORKER.sh):
   - Line 81: Set `RANK=1` (increment for each additional worker: 2, 3, etc.)
   - Line 82: Set `WORLD_SIZE` to match master
   - Line 83: Set `MASTER_ADDR` to the master node's IP address

3. Make the script executable and run:

```bash
chmod +x START_WORKER.sh
./START_WORKER.sh
```

The worker will:
- Auto-detect CUDA if available
- Auto-detect network interface
- Connect to the master node
- Start training

## Configuration Options

### Network Interface Detection

The scripts automatically detect your network interface in this order:
1. `wlan0` (WiFi)
2. `wlp*` (Modern WiFi naming)
3. `eth0` (Ethernet)
4. `enp*` (Modern Ethernet naming)

If detection fails, you can manually set it in the scripts.

### CUDA Configuration

The worker script automatically:
- Detects CUDA installation in `/usr/local/cuda*`
- Sets up environment variables
- Falls back to CPU if CUDA is not available

### Training Parameters

Edit the Python command in both scripts to change:
- `--model`: Model architecture (simple_cnn, resnet50, bert, gpt2)
- `--dataset`: Dataset (synthetic_image, synthetic_text, cifar10)
- `--batch-size`: Batch size per device
- `--epochs`: Number of training epochs
- `--backend`: Communication backend (gloo for CPU/mixed, nccl for GPU-only)

## Troubleshooting

### Connection Issues

If workers can't connect to master:

```bash
# On master, verify port is open
sudo netstat -tlnp | grep 29500

# On worker, test connectivity
telnet <master-ip> 29500
# or
nc -zv <master-ip> 29500

# Check firewall
sudo ufw status
```

### Network Interface Issues

If auto-detection fails:

```bash
# List all interfaces
ip link show

# Manually set in scripts by changing the NETWORK_INTERFACE variable
export GLOO_SOCKET_IFNAME=your_interface_name
export NCCL_SOCKET_IFNAME=your_interface_name
```

### CUDA Issues

If GPU is not detected:

```bash
# Check NVIDIA driver
nvidia-smi

# Check CUDA
nvcc --version

# Verify PyTorch can see GPU
python -c "import torch; print(torch.cuda.is_available())"
```

### Permission Issues

If you get permission errors:

```bash
# Make scripts executable
chmod +x master.sh START_WORKER.sh

# Virtual environment activation
chmod +x venv/bin/activate
```

## Quick Start Example

### Terminal 1 (Master Node):
```bash
cd ~/hetero-cluster
source venv/bin/activate
./master.sh
```

### Terminal 2 (Worker Node):
```bash
cd ~/hetero-cluster
source venv/bin/activate
# Edit START_WORKER.sh to set MASTER_ADDR to master's IP
nano START_WORKER.sh  # or vim
./START_WORKER.sh
```

## Monitoring

During training, you'll see:
- Performance metrics per iteration
- GPU/CPU utilization
- Communication overhead
- Load balancing decisions

Logs are saved to:
- `experiments/<experiment-name>/logs/`
- `experiments/<experiment-name>/metrics/`

## Advanced Configuration

### Multiple Workers

For each additional worker:
1. Clone the repo on the new worker node
2. Set up environment as above
3. Increment `RANK` (worker 2 uses RANK=2, worker 3 uses RANK=3, etc.)
4. Update `WORLD_SIZE` on all nodes to total count
5. Set correct `MASTER_ADDR`

### Mixed CPU/GPU Setup

The system automatically handles heterogeneous hardware:
- Master can be CPU or GPU
- Workers can be CPU or GPU
- Use `--backend gloo` for mixed setups
- Use `--backend nccl` only if all nodes have GPUs

### Load Balancing

Enable adaptive load balancing:
```bash
--enable-load-balancing \
--load-balance-policy dynamic \
--rebalance-interval 10
```

Policies:
- `proportional`: Static allocation based on hardware specs
- `dynamic`: Adaptive based on runtime performance
- `hybrid`: Combination of both

## Next Steps

1. Run the quick test to verify setup
2. Review the dashboard at `http://<master-ip>:3000`
3. Experiment with different models and datasets
4. Check profiling results in the experiments directory

## Support

For issues and questions:
- Check logs in `experiments/*/logs/`
- Review error messages in terminal output
- Ensure network connectivity between nodes
- Verify Python/CUDA versions match requirements
