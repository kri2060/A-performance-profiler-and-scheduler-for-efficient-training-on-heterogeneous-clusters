# Docker Multi-Node Distributed Training Setup

Complete guide for running distributed training across multiple laptops (Windows/Linux) using Docker.

## 🎯 Overview

This setup allows you to train neural networks across multiple machines with **heterogeneous hardware** (GPUs and CPUs) using Docker containers. The system **automatically detects** available hardware and adapts accordingly.

### Why Docker?

- ✅ **Cross-platform**: Works on Windows, Linux, and macOS
- ✅ **Consistent environment**: Same dependencies everywhere
- ✅ **Easy deployment**: No manual Python environment setup
- ✅ **Isolated**: Doesn't interfere with host system
- ✅ **Reproducible**: Same results across all machines
- ✅ **Adaptive**: Automatically uses GPU if available, CPU otherwise

---

## 📋 Prerequisites

### Required for All Machines:

1. **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux)
   - Windows: [Download Docker Desktop](https://www.docker.com/products/docker-desktop/)
   - Linux: `sudo apt-get install docker.io docker-compose`

2. **Network connectivity**: All machines must be on the same network (or accessible via VPN)

3. **Open ports**: Port 29500 must be accessible between machines

### Optional (For GPU Acceleration):

4. **NVIDIA GPU** with proper drivers
   - NVIDIA Driver version >= 450.80.02

5. **NVIDIA Container Toolkit** (for GPU support in Docker)
   ```bash
   # Linux
   distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
   curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
   curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list
   sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
   sudo systemctl restart docker

   # Windows
   # Docker Desktop includes GPU support automatically if NVIDIA drivers are installed
   ```

**Note**: Machines without GPUs can still participate using CPU-only mode!

---

## 🚀 Quick Start Guide

You have two options for deployment:

| Method | Setup Time | When to Use |
|--------|-----------|-------------|
| **Option A: Pull from Registry** | ~1-2 min per machine | ✅ Recommended for most users |
| **Option B: Build Locally** | ~5 min per machine | When you need custom code changes |

### Option A: Pull Pre-built Image (RECOMMENDED - Faster)
### Option B: Build Locally (If you need custom modifications)

---

## 📥 Option A: Using Pre-built Docker Image (Recommended)

This is the fastest way - workers just pull the image instead of building.

### Step 0: Build and Push Image (Master Only - Do This Once)

**On your master laptop**, build and push the image to Docker Hub:

1. **Create a Docker Hub account** (if you don't have one):
   - Go to [hub.docker.com](https://hub.docker.com)
   - Sign up for free

2. **Login to Docker Hub**:
   ```bash
   docker login
   # Enter your Docker Hub username and password
   ```

3. **Build and tag the image**:
   ```bash
   cd /path/to/project
   docker build -t YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest .
   ```
   Replace `YOUR_DOCKERHUB_USERNAME` with your actual Docker Hub username.

4. **Push to Docker Hub**:
   ```bash
   docker push YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest
   ```
   This takes ~2-3 minutes. Now the image is available for all your workers!

5. **Share the image name with your team**:
   ```
   YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest
   ```

### Step 1: Prepare Master Node (Pull Image)

1. **Set the image name**:
   ```bash
   export IMAGE_NAME=YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest
   export WORLD_SIZE=4
   export EXPERIMENT_NAME="multi_node_training"
   ```

2. **Pull the image** (much faster than building):
   ```bash
   docker pull $IMAGE_NAME
   ```

3. **Start the master node**:
   ```bash
   chmod +x scripts/docker_run_master_adaptive.sh
   ./scripts/docker_run_master_adaptive.sh
   ```

   The script will:
   - Auto-detect GPU/CPU
   - Display the master IP and port
   - Wait 10 seconds for workers to connect
   - Start training

### Step 2: Prepare Worker Nodes (Pull Image)

**On Windows:**
```cmd
set IMAGE_NAME=YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest
docker pull %IMAGE_NAME%

set MASTER_ADDR=192.168.1.100
set RANK=1
set WORLD_SIZE=4
scripts\docker_run_worker_adaptive.bat
```

**On Linux:**
```bash
export IMAGE_NAME=YOUR_DOCKERHUB_USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

export MASTER_ADDR=192.168.1.100
export RANK=1
export WORLD_SIZE=4
./scripts/docker_run_worker_adaptive.sh
```

---

## 🔨 Option B: Building Locally

If you want to build on each machine instead of using a registry:

### Step 1: Prepare the Master Node (Your Main Laptop)

1. **Copy the project** to your master laptop:
   ```bash
   cd /path/to/project
   ```

2. **Set environment variables**:
   ```bash
   export WORLD_SIZE=4  # Total number of machines (1 master + 3 workers)
   export EXPERIMENT_NAME="multi_node_training"
   ```

3. **Start the master node**:
   ```bash
   chmod +x scripts/docker_run_master_adaptive.sh
   ./scripts/docker_run_master_adaptive.sh
   ```

   The script will:
   - Build the Docker image (first time only, takes ~5 minutes)
   - Auto-detect GPU/CPU
   - Display the master IP and port
   - Wait 10 seconds for workers to connect
   - Start training

---

### Step 2: Prepare Worker Nodes (Other Laptops - Windows/Linux)

**NEW**: Use the **adaptive scripts** that automatically detect GPU/CPU!

#### On Windows Workers (GPU or CPU):

1. **Copy the entire project folder** to the worker laptop

2. **Open PowerShell or Command Prompt** as Administrator

3. **Navigate to project directory**:
   ```cmd
   cd C:\path\to\project
   ```

4. **Set environment variables** (use the master's IP):
   ```cmd
   set MASTER_ADDR=192.168.1.100
   set MASTER_PORT=29500
   set RANK=1
   set WORLD_SIZE=4
   set EXPERIMENT_NAME=multi_node_training
   ```

   **IMPORTANT**: Each worker must have a **unique RANK**:
   - Worker 1: `set RANK=1`
   - Worker 2: `set RANK=2`
   - Worker 3: `set RANK=3`

5. **Run the adaptive worker script** (auto-detects GPU/CPU):
   ```cmd
   scripts\docker_run_worker_adaptive.bat
   ```

   The script will:
   - ✓ Detect if GPU is available
   - ✓ Use GPU mode if available (higher batch size, GPU acceleration)
   - ✓ Fall back to CPU mode if no GPU (lower batch size, CPU training)

#### On Linux Workers (GPU or CPU):

1. **Copy the project** to the worker laptop

2. **Set environment variables**:
   ```bash
   export MASTER_ADDR=192.168.1.100  # Master's IP
   export MASTER_PORT=29500
   export RANK=1  # Unique for each worker: 1, 2, 3, ...
   export WORLD_SIZE=4
   export EXPERIMENT_NAME=multi_node_training
   ```

3. **Run the adaptive worker script**:
   ```bash
   chmod +x scripts/docker_run_worker_adaptive.sh
   ./scripts/docker_run_worker_adaptive.sh
   ```

#### Legacy Scripts (GPU-only):

If you want to explicitly use GPU mode, use the original scripts:
- Windows: `scripts\docker_run_worker.bat`
- Linux: `scripts/docker_run_worker.sh`

**Note**: These require GPU and will fail on CPU-only machines.

---

### Step 3: Monitor Training (Optional)

**On any machine** (master or worker), run the dashboard:

```bash
# Linux/Mac
./scripts/docker_run_dashboard.sh

# Windows
docker run --rm -it -p 8501:8501 -v "%cd%:/workspace" hetero-cluster-training streamlit run src/monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0
```

**Access dashboard**: Open browser to `http://localhost:8501`

---

## 📊 Example Setups

### Example 1: Hybrid GPU/CPU Setup (4 Laptops)

```
Master Laptop (Linux + GPU):   192.168.1.100:29500  [RANK=0, GPU]
├─ Worker Laptop 1 (Win + GPU): 192.168.1.101        [RANK=1, GPU]
├─ Worker Laptop 2 (Win, CPU):  192.168.1.102        [RANK=2, CPU]
└─ Worker Laptop 3 (Linux,CPU): 192.168.1.103        [RANK=3, CPU]

WORLD_SIZE = 4
```

**Key Features**:
- Master and Worker 1 use GPU acceleration
- Workers 2 and 3 automatically fall back to CPU
- Load balancing adapts to heterogeneous performance
- All workers contribute to training

### Example 2: All-GPU Setup (3 Laptops)

```
Master Laptop (Linux + GPU):    192.168.1.100:29500  [RANK=0, RTX 3070]
├─ Worker Laptop 1 (Win + GPU):  192.168.1.101        [RANK=1, GTX 1660]
└─ Worker Laptop 2 (Linux + GPU):192.168.1.102        [RANK=2, RTX 2060]

WORLD_SIZE = 3
```

### Example 3: All-CPU Setup (2 Laptops)

```
Master Laptop (Linux, CPU):    192.168.1.100:29500  [RANK=0, CPU]
└─ Worker Laptop 1 (Win, CPU):  192.168.1.101        [RANK=1, CPU]

WORLD_SIZE = 2
```

**Note**: CPU-only training is slower but still functional for testing or small models.

### Launch Order

1. **Start Master First** (RANK=0)
2. **Start Workers** within 10 seconds (RANK=1, 2, 3)
3. Training begins automatically
4. System adapts batch sizes and workload based on detected hardware

---

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default | Required |
|----------|-------------|---------|----------|
| `MASTER_ADDR` | Master node's IP address | Auto-detected (master) | Yes (workers) |
| `MASTER_PORT` | Communication port | 29500 | No |
| `RANK` | Node rank (0=master, 1+=workers) | 0 (master) | Yes (workers) |
| `WORLD_SIZE` | Total number of nodes | 4 | Yes |
| `EXPERIMENT_NAME` | Name for saving results | distributed_training | No |

### Customizing Training Parameters

Edit the training command in the scripts (`docker_run_master.sh` or `docker_run_worker.sh`):

```bash
python -m src.training.main \
    --model resnet18 \              # Change model: simple_cnn, resnet18, resnet50
    --dataset cifar10 \             # Change dataset: synthetic_image, cifar10, imagenet
    --batch-size 128 \              # Adjust batch size per GPU
    --epochs 50 \                   # Number of training epochs
    --learning-rate 0.1 \           # Learning rate
    --enable-profiling \            # Enable GPU profiling
    --enable-load-balancing \       # Enable dynamic load balancing
    --load-balance-policy dynamic \ # Policy: static, dynamic, adaptive
    --backend gloo \                # Communication backend: gloo (CPU), nccl (GPU)
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT \
    --experiment-name $EXPERIMENT_NAME
```

---

## 🔄 Alternative: Using Private Registry or File Transfer

If you don't want to use Docker Hub (public registry), you have alternatives:

### Method 1: Save/Load Docker Image (No Internet Required)

**On Master (after building)**:
```bash
# Save image to a tar file
docker save hetero-cluster-training:latest -o hetero-training.tar

# Transfer the file to workers via USB drive, network share, or scp
# File size: ~2-3 GB
scp hetero-training.tar worker@192.168.1.101:~/
```

**On Workers**:
```bash
# Load the image from tar file
docker load -i hetero-training.tar

# Verify it's loaded
docker images | grep hetero-cluster-training
```

### Method 2: Private Docker Registry

Set up your own registry on the master:
```bash
# On master
docker run -d -p 5000:5000 --restart=always --name registry registry:2

# Tag and push
docker tag hetero-cluster-training:latest localhost:5000/hetero-cluster-training:latest
docker push localhost:5000/hetero-cluster-training:latest

# On workers (replace MASTER_IP)
docker pull MASTER_IP:5000/hetero-cluster-training:latest
```

### Method 3: GitHub Container Registry (Free, Private)

```bash
# Login to GitHub
echo $GITHUB_TOKEN | docker login ghcr.io -u YOUR_USERNAME --password-stdin

# Tag and push
docker tag hetero-cluster-training:latest ghcr.io/YOUR_USERNAME/hetero-cluster-training:latest
docker push ghcr.io/YOUR_USERNAME/hetero-cluster-training:latest

# On workers
docker pull ghcr.io/YOUR_USERNAME/hetero-cluster-training:latest
```

---

## 🐛 Troubleshooting

### Issue 1: Workers Can't Connect to Master

**Symptoms**: Workers timeout or can't reach master

**Solutions**:
1. **Check network connectivity**:
   ```bash
   # From worker, ping master
   ping 192.168.1.100
   ```

2. **Check firewall**:
   ```bash
   # Linux - allow port 29500
   sudo ufw allow 29500

   # Windows - add firewall rule in Windows Defender Firewall
   ```

3. **Verify master is listening**:
   ```bash
   # On master
   netstat -an | grep 29500
   ```

### Issue 2: "docker: Error response from daemon: could not select device driver"

**Cause**: NVIDIA Container Toolkit not installed

**Solution**: Install NVIDIA Container Toolkit (see Prerequisites section)

### Issue 3: Docker Image Build Fails

**Cause**: Missing `requirements.txt` or network issues

**Solution**:
```bash
# Check if requirements.txt exists
ls requirements.txt

# Rebuild with no cache
docker build --no-cache -t hetero-cluster-training .
```

### Issue 4: "RuntimeError: Address already in use"

**Cause**: Port 29500 is already in use

**Solution**:
```bash
# Use different port
export MASTER_PORT=29501
```

### Issue 5: GPU Not Detected in Docker

**Solution**:
```bash
# Verify GPU is visible to Docker
docker run --rm --gpus all hetero-cluster-training nvidia-smi

# If this fails, reinstall nvidia-container-toolkit
```

---

## 📁 Directory Structure After Training

```
experiments/
└── {EXPERIMENT_NAME}/
    ├── logs/
    │   ├── rank_0_metrics.json  # Master metrics
    │   ├── rank_1_metrics.json  # Worker 1 metrics
    │   ├── rank_2_metrics.json  # Worker 2 metrics
    │   └── rank_3_metrics.json  # Worker 3 metrics
    ├── configs/
    │   └── gpu_profiles.json    # GPU performance profiles
    └── checkpoints/
        └── model_epoch_*.pth    # Model checkpoints
```

---

## 🔒 Security Notes

1. **Use trusted networks**: Don't expose port 29500 to public internet
2. **VPN recommended**: For training across different locations
3. **Firewall rules**: Only allow specific IPs to connect
4. **Data encryption**: Use VPN for sensitive data

---

## 🎓 Advanced Usage

### Using NCCL Backend (Faster for NVIDIA GPUs)

Replace `--backend gloo` with `--backend nccl` in the scripts:

```bash
--backend nccl \
```

**Note**: NCCL requires all nodes to have NVIDIA GPUs and may not work across Windows/Linux.

### Custom GPU Profiles

Create custom GPU profiles in `experiments/configs/gpu_profiles.json`:

```json
{
  "laptop1_rtx3070": {
    "compute_capability": 8.6,
    "memory_gb": 8,
    "compute_units": 46
  },
  "laptop2_gtx1660": {
    "compute_capability": 7.5,
    "memory_gb": 6,
    "compute_units": 22
  }
}
```

Then use:
```bash
--gpu-profiles experiments/configs/gpu_profiles.json
```

### Training on Single Machine (Multiple GPUs)

Use docker-compose for simpler setup:

```bash
# Edit docker-compose.yml to set WORLD_SIZE
docker-compose up
```

---

## 📞 Getting Help

- **GitHub Issues**: Report bugs or request features
- **Documentation**: Check `README.md` and `QUICKSTART.md`
- **Logs**: Check Docker container logs: `docker logs hetero-master`

---

## 💡 GPU/CPU Hybrid Training Benefits

### Why Use Heterogeneous Hardware?

1. **Utilize All Available Resources**
   - Don't leave CPU-only machines idle
   - Every machine contributes to training speed
   - Maximize cluster utilization

2. **Cost Effective**
   - Use existing hardware without requiring GPUs on all machines
   - Add old laptops to the cluster without additional hardware investment

3. **Flexible Scaling**
   - Start with what you have
   - Add more machines as needed (GPU or CPU)
   - No need for uniform hardware

4. **Automatic Load Balancing**
   - System detects GPU vs CPU performance
   - Adjusts batch sizes automatically (GPU: 64, CPU: 32)
   - Dynamic workload distribution based on processing speed

### Performance Expectations

**Typical Training Speed Contribution**:
- GPU Worker (RTX 3070): 100% baseline
- GPU Worker (GTX 1660): ~60% of baseline
- CPU Worker (8-core): ~10-15% of GPU baseline

**Example 4-Machine Cluster**:
- 2 GPU + 2 CPU ≈ 1.3x faster than 2 GPU alone
- While CPUs are slower, they still provide meaningful speedup

---

## ✅ Checklist

### For GPU Machines:
- [ ] Docker installed
- [ ] NVIDIA drivers installed (>= 450.80.02)
- [ ] NVIDIA Container Toolkit installed
- [ ] GPU detected by `nvidia-smi`

### For CPU-Only Machines:
- [ ] Docker installed (no GPU drivers needed!)

### For All Machines:
- [ ] All machines on same network
- [ ] Port 29500 open/accessible
- [ ] Project copied to all machines
- [ ] `MASTER_ADDR` noted from master node
- [ ] Unique `RANK` set for each worker
- [ ] `WORLD_SIZE` matches total number of machines

---

## 🎉 Success!

If everything is set up correctly, you should see:

**On Master**:
```
Master node ready. Starting training...
Waiting 10 seconds for workers to connect...
Initialized process group for rank 0
Training started on 4 GPUs...
Epoch 1/10: 100%|██████████|
```

**On Workers**:
```
Worker node 1 ready. Connecting to master at 192.168.1.100:29500...
Initialized process group for rank 1
Training started...
```

Happy training! 🚀

---

## 📋 Quick Reference

### One-Time Setup (Choose One)

**Option A: Push to Docker Hub (Recommended)**
```bash
docker login
docker build -t YOUR_USERNAME/hetero-cluster-training:latest .
docker push YOUR_USERNAME/hetero-cluster-training:latest
```

**Option B: Save to File**
```bash
docker build -t hetero-cluster-training:latest .
docker save hetero-cluster-training:latest -o hetero-training.tar
# Transfer file to workers
```

### Master Node

```bash
# If using Docker Hub
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

# Set configuration
export WORLD_SIZE=4
export EXPERIMENT_NAME="my_experiment"

# Start master
./scripts/docker_run_master_adaptive.sh
```

### Worker Nodes

**Linux:**
```bash
# If using Docker Hub
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

# If using saved file
# docker load -i hetero-training.tar

# Configure worker
export MASTER_ADDR=192.168.1.100  # From master output
export RANK=1  # Unique: 1, 2, 3...
export WORLD_SIZE=4

# Start worker
./scripts/docker_run_worker_adaptive.sh
```

**Windows:**
```cmd
REM If using Docker Hub
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull %IMAGE_NAME%

REM Configure worker
set MASTER_ADDR=192.168.1.100
set RANK=1
set WORLD_SIZE=4

REM Start worker
scripts\docker_run_worker_adaptive.bat
```

### Monitor Training

```bash
./scripts/docker_run_dashboard.sh
# Open: http://localhost:8501
```

### Useful Commands

```bash
# Check running containers
docker ps

# View master logs
docker logs hetero-master

# View worker logs
docker logs hetero-worker-1

# Stop all containers
docker stop $(docker ps -q)

# Check Docker images
docker images

# Remove old images
docker image prune -a
```
