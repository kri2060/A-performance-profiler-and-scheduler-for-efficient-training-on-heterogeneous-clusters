# 🐳 Docker Deployment Guide for Worker Machines

Deploy to worker machines using Docker - no need to copy the entire project!

---

## 🎯 Deployment Methods

Choose the best method for your setup:

| Method | Speed | Requirements | Best For |
|--------|-------|--------------|----------|
| **A. Docker Hub** | ⚡ Fastest | Internet, Docker Hub account | Multiple workers |
| **B. Save/Load Image** | 🔄 Medium | USB or network transfer | Offline/Air-gapped |
| **C. Private Registry** | 🚀 Fast | One machine as registry | Local network |

---

## 📦 Method A: Docker Hub (Recommended)

### **Step 1: On Master Machine - Build & Push**

```bash
cd "/home/kri2060/final project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters"

# 1. Login to Docker Hub
docker login
# Enter your username and password

# 2. Tag the image with your Docker Hub username
docker tag hetero-cluster-training:latest YOUR_USERNAME/hetero-cluster-training:latest

# 3. Push to Docker Hub (takes 2-3 minutes)
docker push YOUR_USERNAME/hetero-cluster-training:latest
```

**Share with your team:**
```
Docker Image: YOUR_USERNAME/hetero-cluster-training:latest
Master IP: 10.149.140.68
Master Port: 29500
World Size: 2
```

### **Step 2: On Worker Machine - Pull & Run**

Workers only need Docker installed!

**Linux/macOS Worker:**
```bash
# 1. Pull the image
docker pull YOUR_USERNAME/hetero-cluster-training:latest

# 2. Run worker
docker run --rm \
  --name hetero-worker-1 \
  --network host \
  --gpus all \
  -e MASTER_ADDR=10.149.140.68 \
  -e MASTER_PORT=29500 \
  -e RANK=1 \
  -e WORLD_SIZE=2 \
  YOUR_USERNAME/hetero-cluster-training:latest \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --batch-size 64 \
    --epochs 10 \
    --backend gloo \
    --master-addr 10.149.140.68 \
    --master-port 29500
```

**Windows Worker:**
```cmd
REM Pull the image
docker pull YOUR_USERNAME/hetero-cluster-training:latest

REM Run worker
docker run --rm ^
  --name hetero-worker-1 ^
  --network host ^
  --gpus all ^
  -e MASTER_ADDR=10.149.140.68 ^
  -e MASTER_PORT=29500 ^
  -e RANK=1 ^
  -e WORLD_SIZE=2 ^
  YOUR_USERNAME/hetero-cluster-training:latest ^
  python -m src.training.main ^
    --model simple_cnn ^
    --dataset synthetic_image ^
    --batch-size 64 ^
    --epochs 10 ^
    --backend gloo ^
    --master-addr 10.149.140.68 ^
    --master-port 29500
```

**For CPU-only workers, remove `--gpus all`**

---

## 💾 Method B: Save/Load Docker Image (No Internet Required)

### **Step 1: On Master - Save Image**

```bash
cd "/home/kri2060/final project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters"

# Save Docker image to file
docker save hetero-cluster-training:latest -o hetero-cluster-training.tar

# Check file size
ls -lh hetero-cluster-training.tar
# Expected: ~3-4 GB
```

### **Step 2: Transfer to Worker**

**Option 1: USB Drive**
```bash
# Copy to USB
cp hetero-cluster-training.tar /media/YOUR_USB/

# On worker: Copy from USB
cp /media/YOUR_USB/hetero-cluster-training.tar ~/
```

**Option 2: Network Transfer (SCP)**
```bash
# From master to worker
scp hetero-cluster-training.tar worker@WORKER_IP:~/
```

**Option 3: Already on Your USB!**
Your project folder on the USB already has the image built. You can save it from there!

### **Step 3: On Worker - Load & Run**

```bash
# Load the image
docker load -i hetero-cluster-training.tar

# Verify loaded
docker images | grep hetero-cluster-training

# Run worker
docker run --rm \
  --name hetero-worker-1 \
  --network host \
  --gpus all \
  -e MASTER_ADDR=10.149.140.68 \
  -e MASTER_PORT=29500 \
  -e RANK=1 \
  -e WORLD_SIZE=2 \
  hetero-cluster-training:latest \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --batch-size 64 \
    --epochs 10 \
    --backend gloo \
    --master-addr 10.149.140.68 \
    --master-port 29500
```

---

## 🏢 Method C: Private Docker Registry (Local Network)

Run your own Docker registry on the master machine!

### **Step 1: On Master - Start Registry**

```bash
# Start local Docker registry
docker run -d \
  -p 5000:5000 \
  --restart=always \
  --name registry \
  registry:2

# Tag your image
docker tag hetero-cluster-training:latest localhost:5000/hetero-cluster-training:latest

# Push to local registry
docker push localhost:5000/hetero-cluster-training:latest
```

### **Step 2: On Worker - Pull from Registry**

```bash
# Pull from master's registry
docker pull 10.149.140.68:5000/hetero-cluster-training:latest

# Run worker
docker run --rm \
  --name hetero-worker-1 \
  --network host \
  --gpus all \
  -e MASTER_ADDR=10.149.140.68 \
  -e MASTER_PORT=29500 \
  -e RANK=1 \
  -e WORLD_SIZE=2 \
  10.149.140.68:5000/hetero-cluster-training:latest \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --batch-size 64 \
    --epochs 10 \
    --backend gloo \
    --master-addr 10.149.140.68 \
    --master-port 29500
```

---

## 🚀 Quick Worker Script

Save this as `quick_start_worker.sh` on worker machines:

```bash
#!/bin/bash
# Quick Worker Startup Script

# Configuration - EDIT THESE
DOCKER_IMAGE="YOUR_USERNAME/hetero-cluster-training:latest"
MASTER_ADDR="10.149.140.68"
MASTER_PORT="29500"
RANK=1
WORLD_SIZE=2

# Pull image (comment out if already pulled)
docker pull $DOCKER_IMAGE

# Check for GPU
GPU_FLAG=""
if command -v nvidia-smi &> /dev/null && nvidia-smi &> /dev/null; then
    echo "✓ GPU detected"
    GPU_FLAG="--gpus all"
else
    echo "✓ CPU mode"
fi

# Run worker
docker run --rm \
  --name hetero-worker-$RANK \
  --network host \
  $GPU_FLAG \
  -e MASTER_ADDR=$MASTER_ADDR \
  -e MASTER_PORT=$MASTER_PORT \
  -e RANK=$RANK \
  -e WORLD_SIZE=$WORLD_SIZE \
  $DOCKER_IMAGE \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --batch-size 64 \
    --epochs 10 \
    --backend gloo \
    --master-addr $MASTER_ADDR \
    --master-port $MASTER_PORT
```

**Windows Version** (`quick_start_worker.bat`):

```bat
@echo off
REM Quick Worker Startup Script

REM Configuration - EDIT THESE
set DOCKER_IMAGE=YOUR_USERNAME/hetero-cluster-training:latest
set MASTER_ADDR=10.149.140.68
set MASTER_PORT=29500
set RANK=1
set WORLD_SIZE=2

REM Pull image
docker pull %DOCKER_IMAGE%

REM Check for GPU
nvidia-smi >nul 2>&1
if %errorlevel% == 0 (
    echo GPU detected
    set GPU_FLAG=--gpus all
) else (
    echo CPU mode
    set GPU_FLAG=
)

REM Run worker
docker run --rm ^
  --name hetero-worker-%RANK% ^
  --network host ^
  %GPU_FLAG% ^
  -e MASTER_ADDR=%MASTER_ADDR% ^
  -e MASTER_PORT=%MASTER_PORT% ^
  -e RANK=%RANK% ^
  -e WORLD_SIZE=%WORLD_SIZE% ^
  %DOCKER_IMAGE% ^
  python -m src.training.main ^
    --model simple_cnn ^
    --dataset synthetic_image ^
    --batch-size 64 ^
    --epochs 10 ^
    --backend gloo ^
    --master-addr %MASTER_ADDR% ^
    --master-port %MASTER_PORT%
```

---

## 📋 Worker Machine Requirements

### **Minimal Requirements:**
- ✅ Docker installed
- ✅ Network access to master
- ✅ Port 29500 accessible
- ✅ (Optional) NVIDIA GPU + drivers

### **NO Need For:**
- ❌ Project source code
- ❌ Python environment
- ❌ Dependencies installation
- ❌ Git repository

**Everything is in the Docker image!**

---

## 🔄 Complete Workflow Example

### **Scenario: 1 Master + 2 Workers**

**Master Machine (Your current machine):**
```bash
# 1. Build and push image
docker tag hetero-cluster-training:latest myuser/hetero-training:latest
docker push myuser/hetero-training:latest

# 2. Start master (already running!)
export WORLD_SIZE=3
./scripts/docker_run_master_adaptive.sh
```

**Worker Machine 1:**
```bash
# Just pull and run!
docker pull myuser/hetero-training:latest

docker run --rm --name hetero-worker-1 --network host \
  -e MASTER_ADDR=10.149.140.68 -e RANK=1 -e WORLD_SIZE=3 \
  myuser/hetero-training:latest \
  python -m src.training.main --master-addr 10.149.140.68
```

**Worker Machine 2:**
```bash
docker pull myuser/hetero-training:latest

docker run --rm --name hetero-worker-2 --network host \
  -e MASTER_ADDR=10.149.140.68 -e RANK=2 -e WORLD_SIZE=3 \
  myuser/hetero-training:latest \
  python -m src.training.main --master-addr 10.149.140.68
```

**Done!** Training starts automatically.

---

## 🎯 Advantages of Docker Deployment

### **Benefits:**
- ✅ **Fast deployment** - Just pull image (2 min)
- ✅ **Consistent environment** - Same everywhere
- ✅ **No dependencies** - Everything included
- ✅ **Version control** - Tag different versions
- ✅ **Easy updates** - Just rebuild and push
- ✅ **Platform agnostic** - Works on Linux, Windows, macOS

### **vs Traditional Deployment:**
- ❌ Copy 9GB project folder → ✅ Pull 3GB image
- ❌ Install dependencies → ✅ Already included
- ❌ Setup Python env → ✅ Ready to run
- ❌ Config issues → ✅ Consistent everywhere

---

## 🐛 Troubleshooting

### Issue: "Cannot pull image"

**Docker Hub not accessible:**
```bash
# Use save/load method instead
docker save hetero-cluster-training:latest -o image.tar
# Transfer image.tar to worker
docker load -i image.tar
```

### Issue: "No space left on device"

**Docker images are large:**
```bash
# Clean up old images
docker system prune -a

# Or use external drive
docker save hetero-cluster-training:latest -o /media/usb/image.tar
```

### Issue: Worker can't connect to master

**Check network:**
```bash
# On worker, ping master
ping 10.149.140.68

# Check firewall
# Linux: sudo ufw allow 29500
# Windows: Allow in Windows Firewall
```

---

## 📊 Size Comparison

| Method | Size | Transfer Time (100 Mbps) |
|--------|------|-------------------------|
| Full project | 9 GB | ~12 min |
| Docker image | 3-4 GB | ~5 min |
| Compressed tar | 2-3 GB | ~3 min |

---

## ✅ Recommended Approach

**For 2-4 workers:**
→ Use **Docker Hub** (Method A)

**For air-gapped/offline:**
→ Use **Save/Load** (Method B) with your USB

**For local network cluster:**
→ Use **Private Registry** (Method C)

---

## 🎉 Next Steps

1. **Choose your method** (A, B, or C)
2. **Push/save your image**
3. **Share info with workers**
4. **Workers pull and run**
5. **Training starts!**

Your master is already running and waiting at:
- IP: 10.149.140.68
- Port: 29500
- World Size: 2

Just start a worker using any method above! 🚀
