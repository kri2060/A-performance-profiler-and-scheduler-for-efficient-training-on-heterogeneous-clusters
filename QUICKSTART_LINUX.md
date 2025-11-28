# Quick Start - Linux

## One-Line Setup

```bash
git clone <repo-url> && cd <repo-name> && chmod +x setup_linux.sh && ./setup_linux.sh
```

## Manual Setup (3 Steps)

### Step 1: Clone & Setup
```bash
git clone <repo-url> hetero-cluster
cd hetero-cluster
chmod +x setup_linux.sh
./setup_linux.sh
```

### Step 2: Run Master Node
```bash
# Edit WORLD_SIZE in master.sh (line 48) to total number of nodes
nano master.sh  # or vim

# Run master
./master.sh

# Note the IP address shown in output!
```

### Step 3: Run Worker Node(s)
```bash
# On each worker machine:
# 1. Clone and setup (same as Step 1)
# 2. Edit START_WORKER.sh
nano START_WORKER.sh  # or vim

# Set these values:
# - Line 81: RANK=1 (use 2, 3, etc. for additional workers)
# - Line 82: WORLD_SIZE=2 (must match master)
# - Line 83: MASTER_ADDR=<master-ip-from-step-2>

# Run worker
./START_WORKER.sh
```

## Common Commands

### Check Network Interface
```bash
ip link show
ip addr show wlan0  # or your interface
```

### Get Your IP Address
```bash
hostname -I | awk '{print $1}'
```

### Test Connectivity
```bash
# From worker to master
ping <master-ip>
telnet <master-ip> 29500
```

### Check GPU
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### View Logs
```bash
tail -f experiments/*/logs/rank_*.log
```

### Stop All Training
```bash
pkill -9 -f "python.*src.training.main"
```

## Troubleshooting Quick Fixes

### Can't Connect
```bash
# Check firewall
sudo ufw status
sudo ufw allow 29500/tcp

# Or disable temporarily
sudo ufw disable
```

### Wrong Network Interface
```bash
# Edit scripts and manually set:
export GLOO_SOCKET_IFNAME=your_interface_name
export NCCL_SOCKET_IFNAME=your_interface_name
```

### Python/CUDA Issues
```bash
# Reinstall PyTorch
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# For GPU support:
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Default Configuration

- **Backend**: Gloo (works with CPU and GPU)
- **Port**: 29500
- **Model**: SimpleCNN
- **Dataset**: Synthetic (1000 samples)
- **Batch Size**: 32
- **Epochs**: 5

## File Structure

```
.
├── master.sh              # Run on master node
├── START_WORKER.sh        # Run on worker nodes
├── setup_linux.sh         # Automated setup script
├── LINUX_SETUP.md         # Detailed setup guide
└── src/
    ├── training/          # Training code
    ├── profiling/         # Performance profiling
    ├── scheduling/        # Load balancing
    └── utils/             # Utilities
```

## Next Steps

1. Run quick test to verify setup
2. Check [LINUX_SETUP.md](LINUX_SETUP.md) for detailed configuration
3. Experiment with different models and datasets
4. Review profiling results in `experiments/` directory

## Support

- Detailed setup: [LINUX_SETUP.md](LINUX_SETUP.md)
- Check logs: `experiments/*/logs/`
- Test connectivity between nodes
- Verify Python/CUDA versions
