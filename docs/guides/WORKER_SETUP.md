# Worker Node Setup Guide

This guide walks you through setting up **worker nodes** (RANK 1, 2, 3, ...) for distributed training.

---

## Prerequisites

✅ Complete [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) first
✅ Master node is running (see [MASTER_SETUP.md](MASTER_SETUP.md))
✅ You have the master's IP address

---

## Information You Need from Master

Get this information from the person running the master node:

```
Master IP:    ________________  (e.g., 192.168.1.100)
Master Port:  ________________  (usually 29500)
World Size:   ________________  (e.g., 4)
Your Rank:    ________________  (1, 2, 3, etc. - MUST BE UNIQUE!)
Image Name:   ________________  (if using Docker Hub)
```

---

## Important: Unique RANK Assignment

**Each worker MUST have a unique RANK:**

```
Master:   RANK=0 (already running)
Worker 1: RANK=1  ← You might be this
Worker 2: RANK=2
Worker 3: RANK=3
...
```

⚠️ **Never duplicate RANKs** - this will cause training to fail!

---

## Setup Instructions by Operating System

Choose your OS:
- [Linux Worker Setup](#linux-worker-setup)
- [Windows Worker Setup](#windows-worker-setup)
- [macOS Worker Setup](#macos-worker-setup)

---

## Linux Worker Setup

### Method A: Using Docker Hub Image (Recommended)

#### 1. Pull Docker Image

```bash
# Set image name (get this from master setup)
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

# Pull image (faster than building)
docker pull $IMAGE_NAME
```

#### 2. Configure Environment

```bash
# Get from master
export MASTER_ADDR=192.168.1.100  # Master's IP
export MASTER_PORT=29500           # Usually 29500
export WORLD_SIZE=4                # Total machines

# YOUR unique rank (1, 2, 3, ...)
export RANK=1  # ⚠️ CHANGE THIS - must be unique!

export EXPERIMENT_NAME=multi_node_training
```

#### 3. Test Connection to Master

```bash
# Test if master is reachable
ping -c 3 $MASTER_ADDR

# Test if port is accessible (after master starts)
telnet $MASTER_ADDR $MASTER_PORT
# Press Ctrl+] then type 'quit' to exit
```

#### 4. Start Worker

```bash
# Navigate to project (if using local scripts)
cd /path/to/project

# Make script executable
chmod +x scripts/docker_run_worker_adaptive.sh

# Start worker
./scripts/docker_run_worker_adaptive.sh
```

**Expected Output:**
```
🚀 Starting worker node with adaptive GPU/CPU detection...

🔍 Detecting hardware...
✓ GPU detected: NVIDIA GTX 1660
✓ Using GPU mode with batch size 64

📡 Network configuration:
   Master: 192.168.1.100:29500
   Rank: 1
   World Size: 4

🔗 Connecting to master...
[Worker container started]
```

---

### Method B: Build Locally

#### 1. Copy Project to Worker

```bash
# Option 1: Git clone (if you have repo)
git clone <your-repo-url>
cd A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Option 2: SCP from master
scp -r user@master_ip:/path/to/project ~/project
cd ~/project
```

#### 2. Configure and Start

```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=1  # Unique rank
export WORLD_SIZE=4
export EXPERIMENT_NAME=multi_node_training

# Start worker (will build on first run)
chmod +x scripts/docker_run_worker_adaptive.sh
./scripts/docker_run_worker_adaptive.sh
```

---

### Method C: Load from Saved Image

#### 1. Load Docker Image

```bash
# If you received hetero-training.tar file
docker load -i /path/to/hetero-training.tar

# Verify image loaded
docker images | grep hetero-cluster-training
```

#### 2. Configure and Start

```bash
export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=1
export WORLD_SIZE=4
export EXPERIMENT_NAME=multi_node_training

chmod +x scripts/docker_run_worker_adaptive.sh
./scripts/docker_run_worker_adaptive.sh
```

---

## Windows Worker Setup

### Method A: Using Docker Hub Image (Recommended)

#### 1. Open PowerShell or Command Prompt

Run as Administrator.

#### 2. Navigate to Project Directory

```cmd
cd C:\path\to\project
```

Or if project not available, just use Docker directly:

```cmd
mkdir C:\hetero-training
cd C:\hetero-training
```

#### 3. Pull Docker Image

```cmd
REM Set image name
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

REM Pull image
docker pull %IMAGE_NAME%
```

#### 4. Configure Environment Variables

```cmd
REM Get from master
set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set WORLD_SIZE=4

REM YOUR unique rank
set RANK=1

set EXPERIMENT_NAME=multi_node_training
```

#### 5. Test Connection

```cmd
REM Test ping
ping %MASTER_ADDR%

REM Test port (after master starts)
Test-NetConnection -ComputerName %MASTER_ADDR% -Port %MASTER_PORT%
```

#### 6. Start Worker

**Option 1: Using provided script (if you have project files)**
```cmd
scripts\docker_run_worker_adaptive.bat
```

**Option 2: Direct Docker command**
```cmd
docker run --rm ^
  --name hetero-worker-%RANK% ^
  --network host ^
  -v "%cd%:/workspace" ^
  -e MASTER_ADDR=%MASTER_ADDR% ^
  -e MASTER_PORT=%MASTER_PORT% ^
  -e RANK=%RANK% ^
  -e WORLD_SIZE=%WORLD_SIZE% ^
  -e EXPERIMENT_NAME=%EXPERIMENT_NAME% ^
  %IMAGE_NAME% ^
  python -m src.training.main --master-addr %MASTER_ADDR% --master-port %MASTER_PORT%
```

**For GPU machines, add:**
```cmd
--gpus all ^
```

---

### Method B: Build Locally (Windows)

#### 1. Copy Project Files

Copy the entire project folder to your Windows machine.

#### 2. Open PowerShell as Admin

```powershell
cd C:\path\to\project
```

#### 3. Set Environment Variables

```cmd
set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set RANK=1
set WORLD_SIZE=4
set EXPERIMENT_NAME=multi_node_training
```

#### 4. Run Worker Script

```cmd
scripts\docker_run_worker_adaptive.bat
```

First run will build the image (~5 minutes).

---

### Method C: Load Saved Image (Windows)

#### 1. Load Image File

```cmd
REM Load from tar file
docker load -i C:\path\to\hetero-training.tar

REM Verify
docker images
```

#### 2. Configure and Start

```cmd
set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set RANK=1
set WORLD_SIZE=4

scripts\docker_run_worker_adaptive.bat
```

---

## macOS Worker Setup

Similar to Linux, use the Linux commands but with macOS-specific IP detection:

```bash
# Get IP address on macOS
ipconfig getifaddr en0

# Everything else same as Linux
export MASTER_ADDR=192.168.1.100
export RANK=1
export WORLD_SIZE=4

./scripts/docker_run_worker_adaptive.sh
```

---

## Verify Worker is Running

### Check Container Status

```bash
# Linux/macOS
docker ps

# Windows
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                          STATUS
xyz789abc123   hetero-cluster-training:latest Up 1 minute
```

### Check Worker Logs

```bash
# Linux/macOS
docker logs hetero-worker-1

# Windows
docker logs hetero-worker-1
```

**Expected Output:**
```
[Worker] Rank 1 connecting to master at 192.168.1.100:29500
[Worker] Successfully connected to master
[Worker] Initialized process group
[Worker] Device: cuda:0
[Worker] Starting training...
```

---

## Troubleshooting

### Issue: "Cannot connect to master"

**Check network connectivity:**
```bash
ping 192.168.1.100
```

**Check master is running:**
```bash
# On master machine
docker ps | grep hetero-master
```

**Check firewall:**
```bash
# Linux - allow port
sudo ufw allow 29500

# Windows - add firewall rule
# Windows Defender Firewall → Inbound Rules → New Rule → Port 29500
```

### Issue: "RuntimeError: Address already in use"

**You're using a duplicate RANK!**

Each worker needs a unique RANK. Check with your team:
- Worker 1: RANK=1
- Worker 2: RANK=2
- Worker 3: RANK=3

### Issue: "Connection timeout"

**Master might not be ready:**
- Wait for master to fully start
- Master needs 60 seconds to initialize
- Check master logs: `docker logs hetero-master`

### Issue: "Docker: command not found"

**Install Docker:**
```bash
# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Windows: Download Docker Desktop from docker.com
```

### Issue: GPU not detected (but you have GPU)

**Check NVIDIA drivers:**
```bash
nvidia-smi
```

**Check Docker GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

**Install NVIDIA Container Toolkit** (Linux only):
```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
  sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

### Issue: "No such file or directory: scripts/docker_run_worker_adaptive.sh"

**You don't have project files. Two options:**

**Option 1: Get project files**
```bash
git clone <repo-url>
cd project
```

**Option 2: Run Docker directly**
```bash
docker run --rm \
  --name hetero-worker-1 \
  --network host \
  -e MASTER_ADDR=192.168.1.100 \
  -e MASTER_PORT=29500 \
  -e RANK=1 \
  -e WORLD_SIZE=4 \
  YOUR_USERNAME/hetero-cluster-training:latest \
  python -m src.training.main --master-addr $MASTER_ADDR
```

---

## CPU vs GPU Mode

The adaptive scripts automatically detect your hardware:

### GPU Detected:
```
✓ GPU detected: NVIDIA RTX 3070
✓ Using GPU mode
✓ Batch size: 64
✓ Backend: NCCL (if all nodes have GPU)
```

### CPU Only:
```
ℹ No GPU detected
✓ Using CPU mode
✓ Batch size: 32 (lower for CPU)
✓ Backend: Gloo
```

Both modes work! CPU machines contribute to training, just slower than GPU.

---

## Monitoring Worker Progress

### Real-time Logs

```bash
# Follow logs
docker logs -f hetero-worker-1
```

### Check Training Metrics

```bash
# On worker machine
ls experiments/multi_node_training/logs/

# View your worker's metrics
cat experiments/multi_node_training/logs/rank_1_metrics.json
```

### Access Worker Container

```bash
docker exec -it hetero-worker-1 /bin/bash
```

---

## Multiple Workers on Same Machine

You can run multiple workers on one machine (if you have multiple GPUs):

```bash
# Worker 1 - GPU 0
export RANK=1
export CUDA_VISIBLE_DEVICES=0
./scripts/docker_run_worker_adaptive.sh

# Worker 2 - GPU 1
export RANK=2
export CUDA_VISIBLE_DEVICES=1
./scripts/docker_run_worker_adaptive.sh
```

---

## Success Indicators

✅ Worker connected successfully if you see:

1. Container running: `docker ps` shows worker container
2. Logs show: "Successfully connected to master"
3. Logs show: "Initialized process group for rank X"
4. Logs show: "Starting training..."
5. No connection errors

---

## Stopping Worker

```bash
# Stop container
docker stop hetero-worker-1

# Remove container
docker rm hetero-worker-1

# Verify stopped
docker ps
```

---

## Quick Reference Card

```bash
# ============================================
# Worker Quick Start (Linux/macOS)
# ============================================

# 1. Pull image
export IMAGE_NAME=USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

# 2. Configure
export MASTER_ADDR=192.168.1.100  # From master
export MASTER_PORT=29500
export RANK=1                      # Your unique rank
export WORLD_SIZE=4                # Total machines

# 3. Test connection
ping $MASTER_ADDR

# 4. Start worker
./scripts/docker_run_worker_adaptive.sh

# 5. Monitor
docker logs -f hetero-worker-1
```

```cmd
REM ============================================
REM Worker Quick Start (Windows)
REM ============================================

REM 1. Pull image
set IMAGE_NAME=USERNAME/hetero-cluster-training:latest
docker pull %IMAGE_NAME%

REM 2. Configure
set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set RANK=1
set WORLD_SIZE=4

REM 3. Test connection
ping %MASTER_ADDR%

REM 4. Start worker
scripts\docker_run_worker_adaptive.bat

REM 5. Monitor
docker logs -f hetero-worker-1
```

---

## Next Steps

1. ✅ Worker is running
2. ✅ Connected to master
3. ⏭️  **Monitor training** → See [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)
4. ⏭️  **View dashboard** → `http://MASTER_IP:8501`

Happy distributed training! 🚀
