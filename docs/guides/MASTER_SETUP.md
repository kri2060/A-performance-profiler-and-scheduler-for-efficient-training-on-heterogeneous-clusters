# Master Node Setup Guide

This guide walks you through setting up the **master node** (RANK=0) for distributed training.

---

## Prerequisites

✅ Complete [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md) first

---

## Step-by-Step Setup

### Step 1: Choose Deployment Method

Pick ONE method based on your needs:

| Method | Setup Time | Best For |
|--------|-----------|----------|
| **A. Docker Hub** | 2 min | Multiple workers, fastest deployment |
| **B. Build Locally** | 5 min | Single machine testing, custom code |
| **C. Save/Load** | 3 min + transfer | No internet on workers |

---

## Method A: Docker Hub (Recommended for Multi-Machine)

### 1. Create Docker Hub Account (One-time)

1. Go to [hub.docker.com](https://hub.docker.com)
2. Sign up for free account
3. Note your username: `YOUR_USERNAME`

### 2. Login to Docker Hub

```bash
docker login
# Enter username and password
```

### 3. Build and Push Image (Master Only)

```bash
cd /home/kri2060/final\ project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Build image
docker build -t YOUR_USERNAME/hetero-cluster-training:latest .

# Push to Docker Hub (takes 2-3 minutes)
docker push YOUR_USERNAME/hetero-cluster-training:latest
```

**Share this with your team:**
```
Image Name: YOUR_USERNAME/hetero-cluster-training:latest
```

### 4. Configure Master Environment

```bash
# Set deployment configuration
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
export WORLD_SIZE=4  # Total machines: 1 master + 3 workers
export EXPERIMENT_NAME="multi_node_training"

# Pull the image
docker pull $IMAGE_NAME
```

### 5. Get Master IP Address

```bash
# Linux
hostname -I | awk '{print $1}'

# macOS
ipconfig getifaddr en0

# Windows (PowerShell)
(Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*"}).IPAddress
```

**Note this IP address** - you'll share it with workers.

Example: `192.168.1.100`

### 6. Start Master Node

```bash
chmod +x scripts/docker_run_master_adaptive.sh
./scripts/docker_run_master_adaptive.sh
```

**Expected Output:**
```
🚀 Starting master node with adaptive GPU/CPU detection...

🔍 Detecting hardware...
✓ GPU detected: NVIDIA GeForce RTX 3070
✓ Using GPU mode with batch size 64

📡 Network configuration:
   Master IP: 192.168.1.100
   Master Port: 29500
   Rank: 0
   World Size: 4

⏳ Waiting 60 seconds for workers to connect...
   Workers should connect to: 192.168.1.100:29500

[Master container started]
```

**Important:**
- Share the Master IP (192.168.1.100) with all workers
- Workers must connect within 60 seconds
- Don't kill the process - leave it running

---

## Method B: Build Locally (Simpler, No Registry)

### 1. Navigate to Project

```bash
cd /home/kri2060/final\ project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters
```

### 2. Configure Environment

```bash
export WORLD_SIZE=4  # 1 master + 3 workers
export EXPERIMENT_NAME="multi_node_training"
```

### 3. Get Master IP

```bash
# Linux
hostname -I | awk '{print $1}'

# Output example: 192.168.1.100
```

### 4. Start Master (Will Build Automatically)

```bash
chmod +x scripts/docker_run_master_adaptive.sh
./scripts/docker_run_master_adaptive.sh
```

First run will build the Docker image (~5 minutes). Subsequent runs will be fast.

---

## Method C: Save/Load Image (Offline/Air-gapped)

### 1. Build Image

```bash
cd /home/kri2060/final\ project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

docker build -t hetero-cluster-training:latest .
```

### 2. Save Image to File

```bash
# Save to tar file (~3GB)
docker save hetero-cluster-training:latest -o hetero-training.tar

# Verify file created
ls -lh hetero-training.tar
```

### 3. Transfer to Workers

Choose one method:

**Option 1: USB Drive**
```bash
# Copy to USB
cp hetero-training.tar /media/usb/

# On worker: Copy from USB to home directory
```

**Option 2: Network Transfer (SCP)**
```bash
# Transfer to each worker
scp hetero-training.tar worker1@192.168.1.101:~/
scp hetero-training.tar worker2@192.168.1.102:~/
scp hetero-training.tar worker3@192.168.1.103:~/
```

**Option 3: Shared Network Drive**
```bash
# Copy to shared drive accessible by all machines
```

### 4. Start Master

```bash
export WORLD_SIZE=4
export EXPERIMENT_NAME="multi_node_training"

chmod +x scripts/docker_run_master_adaptive.sh
./scripts/docker_run_master_adaptive.sh
```

---

## Verify Master is Running

### Check Container Status

```bash
docker ps
```

**Expected Output:**
```
CONTAINER ID   IMAGE                          STATUS         PORTS
abc123def456   hetero-cluster-training:latest Up 2 minutes   0.0.0.0:29500->29500/tcp
```

### Check Logs

```bash
docker logs hetero-master
```

**Expected Output:**
```
[Master] Initializing distributed training...
[Master] Waiting for workers to connect...
[Master] Rank 0 initialized on device cuda:0
```

### Check Network Connectivity

```bash
# Check if port 29500 is listening
netstat -tuln | grep 29500

# Expected output:
tcp        0      0 0.0.0.0:29500           0.0.0.0:*               LISTEN
```

---

## Common Issues

### Issue: "Address already in use"

**Solution:** Port 29500 is taken
```bash
# Use different port
export MASTER_PORT=29501

# Update workers too!
```

### Issue: "Docker daemon not running"

**Solution:**
```bash
# Linux
sudo systemctl start docker

# Windows/Mac: Start Docker Desktop
```

### Issue: "Cannot find image"

**Solution:** Build or pull image first
```bash
# For Docker Hub
docker pull YOUR_USERNAME/hetero-cluster-training:latest

# For local build
docker build -t hetero-cluster-training:latest .
```

### Issue: GPU not detected

**Check GPU:**
```bash
nvidia-smi
```

**Check Docker GPU access:**
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

If this fails, reinstall NVIDIA Container Toolkit.

---

## Environment Variables Reference

| Variable | Description | Example | Required |
|----------|-------------|---------|----------|
| `WORLD_SIZE` | Total number of machines | `4` | Yes |
| `EXPERIMENT_NAME` | Name for results | `multi_node_training` | No |
| `MASTER_PORT` | Communication port | `29500` | No (default) |
| `IMAGE_NAME` | Docker image to use | `user/image:latest` | Only for Method A |

---

## What Information to Share with Workers

Once master is running, share this info with your team:

```
=================================
🎯 Master Node Information
=================================
Master IP:    192.168.1.100
Master Port:  29500
World Size:   4

Docker Image: YOUR_USERNAME/hetero-cluster-training:latest
  (or method you're using)

Workers needed: 3 (RANK 1, 2, 3)
=================================
```

---

## Next Steps

1. ✅ Master is running
2. ⏭️  **Set up workers** → See [WORKER_SETUP.md](WORKER_SETUP.md)
3. ⏭️  **Launch training** → See [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)
4. ⏭️  **Monitor progress** → Dashboard at `http://localhost:8501`

---

## Stopping the Master

```bash
# Stop container
docker stop hetero-master

# Remove container
docker rm hetero-master

# Check no containers running
docker ps
```

---

## Advanced Configuration

### Custom Training Parameters

Edit [scripts/docker_run_master_adaptive.sh](scripts/docker_run_master_adaptive.sh):

```bash
python -m src.training.main \
    --model resnet50 \              # Change model
    --dataset cifar10 \             # Change dataset
    --batch-size 128 \              # Adjust batch size
    --epochs 100 \                  # Number of epochs
    --learning-rate 0.1 \           # Learning rate
    --enable-profiling \            # Enable GPU profiling
    --enable-load-balancing \       # Enable adaptive balancing
    --load-balance-policy dynamic   # Balancing policy
```

### Using NCCL (Faster for NVIDIA GPUs)

If all machines have NVIDIA GPUs:

```bash
# In training command, change:
--backend nccl
```

**Note:** NCCL requires all nodes have GPUs and may not work across Windows/Linux.

---

## Monitoring

### View Master Logs (Real-time)

```bash
docker logs -f hetero-master
```

### Access Container Shell

```bash
docker exec -it hetero-master /bin/bash
```

### Check Training Metrics

```bash
# View experiment results
ls experiments/multi_node_training/

# View metrics
cat experiments/multi_node_training/logs/rank_0_metrics.json
```

---

## Success Indicators

✅ Master started successfully if you see:

1. Container running: `docker ps` shows hetero-master
2. Port listening: `netstat -tuln | grep 29500`
3. Logs show: "Waiting for workers to connect"
4. No errors in: `docker logs hetero-master`

Ready for workers! 🚀
