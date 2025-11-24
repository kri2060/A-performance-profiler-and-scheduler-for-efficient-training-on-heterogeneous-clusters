# Launch Guide - Complete Training Examples

This guide provides complete, step-by-step examples for launching distributed training across multiple machines.

---

## Prerequisites

✅ Master node setup complete ([MASTER_SETUP.md](MASTER_SETUP.md))
✅ All worker nodes setup complete ([WORKER_SETUP.md](WORKER_SETUP.md))
✅ All machines can communicate with master

---

## Complete Example Scenarios

Choose the scenario that matches your setup:

1. [4-Machine Setup (2 GPU + 2 CPU)](#scenario-1-4-machine-hybrid-setup)
2. [3-Machine All-GPU Setup](#scenario-2-3-machine-all-gpu-setup)
3. [2-Machine CPU-Only Setup](#scenario-3-2-machine-cpu-only-setup)
4. [Mixed Windows/Linux Setup](#scenario-4-mixed-windowslinux-setup)

---

## Scenario 1: 4-Machine Hybrid Setup

**Configuration:**
```
Machine 1 (Master): Linux + RTX 3070    → RANK=0 @ 192.168.1.100
Machine 2 (Worker): Windows + GTX 1660 → RANK=1 @ 192.168.1.101
Machine 3 (Worker): Linux, CPU only    → RANK=2 @ 192.168.1.102
Machine 4 (Worker): Windows, CPU only  → RANK=3 @ 192.168.1.103

WORLD_SIZE = 4
```

### Step 1: Start Master (Machine 1 - Linux)

```bash
cd /home/kri2060/final\ project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Set configuration
export WORLD_SIZE=4
export EXPERIMENT_NAME="hybrid_4node_training"

# Option A: Using Docker Hub
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

# Start master
chmod +x scripts/docker_run_master_adaptive.sh
./scripts/docker_run_master_adaptive.sh
```

**Wait for output:**
```
✓ Master IP: 192.168.1.100
⏳ Waiting 60 seconds for workers to connect...
```

**Share with team:**
```
Master IP: 192.168.1.100
World Size: 4
Image: YOUR_USERNAME/hetero-cluster-training:latest
```

### Step 2: Start Worker 1 (Machine 2 - Windows GPU)

Open PowerShell as Admin:

```cmd
REM Navigate to project or create working directory
cd C:\hetero-training

REM Pull image
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull %IMAGE_NAME%

REM Configure
set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set RANK=1
set WORLD_SIZE=4
set EXPERIMENT_NAME=hybrid_4node_training

REM Test connection
ping %MASTER_ADDR%

REM Start worker (GPU will be auto-detected)
scripts\docker_run_worker_adaptive.bat
```

### Step 3: Start Worker 2 (Machine 3 - Linux CPU)

```bash
cd ~/project

# Configure
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull $IMAGE_NAME

export MASTER_ADDR=192.168.1.100
export MASTER_PORT=29500
export RANK=2
export WORLD_SIZE=4
export EXPERIMENT_NAME=hybrid_4node_training

# Test connection
ping -c 3 $MASTER_ADDR

# Start worker (will detect CPU-only)
chmod +x scripts/docker_run_worker_adaptive.sh
./scripts/docker_run_worker_adaptive.sh
```

### Step 4: Start Worker 3 (Machine 4 - Windows CPU)

Open PowerShell as Admin:

```cmd
cd C:\hetero-training

set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest
docker pull %IMAGE_NAME%

set MASTER_ADDR=192.168.1.100
set MASTER_PORT=29500
set RANK=3
set WORLD_SIZE=4
set EXPERIMENT_NAME=hybrid_4node_training

ping %MASTER_ADDR%

scripts\docker_run_worker_adaptive.bat
```

### Step 5: Monitor Training

**On Master machine:**

```bash
# Watch master logs
docker logs -f hetero-master

# Start dashboard (optional)
./scripts/docker_run_dashboard.sh
# Open: http://192.168.1.100:8501
```

**Expected Training Output:**

```
[Rank 0] Master ready. All workers connected!
[Rank 1] Worker connected (GPU: GTX 1660)
[Rank 2] Worker connected (CPU mode)
[Rank 3] Worker connected (CPU mode)

Epoch 1/10:
  Rank 0 (RTX 3070):  Batch size=64,  Speed=256 samples/s
  Rank 1 (GTX 1660):  Batch size=64,  Speed=180 samples/s
  Rank 2 (CPU):       Batch size=32,  Speed=45 samples/s
  Rank 3 (CPU):       Batch size=32,  Speed=40 samples/s

Load Balancer: Adjusting workload distribution...
Throughput: 521 samples/s (cluster total)
```

---

## Scenario 2: 3-Machine All-GPU Setup

**Configuration:**
```
Machine 1 (Master): Linux + RTX 3070 → RANK=0 @ 192.168.1.100
Machine 2 (Worker): Linux + RTX 2060 → RANK=1 @ 192.168.1.101
Machine 3 (Worker): Win + GTX 1660   → RANK=2 @ 192.168.1.102

WORLD_SIZE = 3
Backend: NCCL (faster for all-GPU)
```

### Master (Machine 1):

```bash
export WORLD_SIZE=3
export EXPERIMENT_NAME="gpu_cluster_training"
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull $IMAGE_NAME

# Use NCCL backend for better GPU performance
# Edit scripts/docker_run_master_adaptive.sh and change:
# --backend gloo  →  --backend nccl

./scripts/docker_run_master_adaptive.sh
```

### Worker 1 (Machine 2 - Linux):

```bash
export MASTER_ADDR=192.168.1.100
export RANK=1
export WORLD_SIZE=3
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull $IMAGE_NAME
./scripts/docker_run_worker_adaptive.sh
```

### Worker 2 (Machine 3 - Windows):

```cmd
set MASTER_ADDR=192.168.1.100
set RANK=2
set WORLD_SIZE=3
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull %IMAGE_NAME%
scripts\docker_run_worker_adaptive.bat
```

**Expected Performance:**
```
All GPUs detected - using NCCL backend
Higher throughput due to GPU acceleration
Automatic load balancing based on GPU performance
```

---

## Scenario 3: 2-Machine CPU-Only Setup

**Configuration:**
```
Machine 1 (Master): Linux CPU → RANK=0 @ 192.168.1.100
Machine 2 (Worker): Win CPU   → RANK=1 @ 192.168.1.101

WORLD_SIZE = 2
```

Perfect for testing without GPUs!

### Master (Machine 1):

```bash
export WORLD_SIZE=2
export EXPERIMENT_NAME="cpu_test"
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull $IMAGE_NAME
./scripts/docker_run_master_adaptive.sh

# Will automatically use:
# - CPU mode
# - Gloo backend
# - Lower batch size (32)
```

### Worker (Machine 2):

```cmd
set MASTER_ADDR=192.168.1.100
set RANK=1
set WORLD_SIZE=2
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull %IMAGE_NAME%
scripts\docker_run_worker_adaptive.bat
```

**Note:** CPU training is slower but functional for testing and small models.

---

## Scenario 4: Mixed Windows/Linux Setup

**Configuration:**
```
Machine 1 (Master): Windows + GPU → RANK=0 @ 192.168.1.50
Machine 2 (Worker): Linux + GPU   → RANK=1 @ 192.168.1.51
Machine 3 (Worker): Linux, CPU    → RANK=2 @ 192.168.1.52

WORLD_SIZE = 3
```

### Master (Windows):

```cmd
cd C:\project

set WORLD_SIZE=3
set EXPERIMENT_NAME=mixed_os_training
set IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull %IMAGE_NAME%

REM Find your IP
ipconfig | findstr IPv4

REM Start master
scripts\docker_run_master_adaptive.bat
```

### Worker 1 (Linux GPU):

```bash
export MASTER_ADDR=192.168.1.50  # Windows master IP
export RANK=1
export WORLD_SIZE=3
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull $IMAGE_NAME
./scripts/docker_run_worker_adaptive.sh
```

### Worker 2 (Linux CPU):

```bash
export MASTER_ADDR=192.168.1.50
export RANK=2
export WORLD_SIZE=3
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster-training:latest

docker pull $IMAGE_NAME
./scripts/docker_run_worker_adaptive.sh
```

---

## Launch Sequence Best Practices

### ⏱️ Timing

1. **Start Master first** (wait for "Waiting for workers..." message)
2. **Start workers within 60 seconds**
3. Workers can start in any order
4. Training begins automatically when all workers connect

### 🔄 Recommended Order

```
1. Master starts       → Waits 60 seconds
2. Worker 1 starts     → Connects immediately
3. Worker 2 starts     → Connects immediately
4. Worker 3 starts     → Connects immediately
5. Training begins     → All nodes synchronized
```

### ⚠️ What If Workers Are Late?

**If workers don't connect within 60 seconds:**
```bash
# Master will show timeout
# Solution: Restart master with longer timeout

# Edit scripts/docker_run_master_adaptive.sh
# Change: sleep 60  →  sleep 120
```

---

## Monitoring and Dashboards

### Option 1: Command Line Monitoring

**On Master:**
```bash
# Real-time logs
docker logs -f hetero-master

# Watch GPU usage (if GPU)
watch -n 1 nvidia-smi
```

**On Workers:**
```bash
docker logs -f hetero-worker-1
```

### Option 2: Web Dashboard

**Start Streamlit dashboard:**

```bash
# On master or any machine with access to shared storage
./scripts/docker_run_dashboard.sh
```

**Access dashboard:**
```
http://MASTER_IP:8501
```

**Features:**
- Real-time throughput graphs
- Per-rank performance metrics
- GPU utilization (if available)
- Load balancing decisions
- Training progress

### Option 3: Logs and Metrics Files

**Check experiment results:**
```bash
ls experiments/EXPERIMENT_NAME/

# Metrics per rank
cat experiments/EXPERIMENT_NAME/logs/rank_0_metrics.json
cat experiments/EXPERIMENT_NAME/logs/rank_1_metrics.json
```

---

## Training Parameters Customization

### Changing Model and Dataset

Edit the training command in scripts:

**Simple CNN (fast testing):**
```bash
python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --num-samples 5000 \
    --epochs 10
```

**ResNet-18 with CIFAR-10:**
```bash
python -m src.training.main \
    --model resnet18 \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 50
```

**ResNet-50 (larger model):**
```bash
python -m src.training.main \
    --model resnet50 \
    --dataset cifar10 \
    --batch-size 64 \
    --epochs 100
```

### Load Balancing Policies

**Static (baseline):**
```bash
--load-balance-policy static
```
All ranks use same batch size.

**Dynamic (adaptive):**
```bash
--enable-load-balancing \
--load-balance-policy dynamic
```
Adjusts batch sizes based on GPU performance.

**Proportional:**
```bash
--enable-load-balancing \
--load-balance-policy proportional
```
Batch sizes proportional to GPU compute capability.

### Communication Backends

**Gloo (CPU and cross-platform):**
```bash
--backend gloo
```
Works everywhere, slower for GPUs.

**NCCL (NVIDIA GPUs only):**
```bash
--backend nccl
```
Faster for all-GPU clusters.

---

## Progress Tracking

### What to Expect During Training

**Initialization (0-30 seconds):**
```
[Master] Starting master node...
[Worker 1] Connecting to master...
[Worker 2] Connecting to master...
[All] Process group initialized
```

**Training Loop:**
```
Epoch 1/10:
  Rank 0 | Batch 0/100 | Loss: 2.3045
  Rank 1 | Batch 0/50  | Loss: 2.3102
  ...

Epoch 1 completed in 120.5s
Average throughput: 450 samples/s
```

**Load Balancing (if enabled):**
```
Load Balancer: Detected straggler on Rank 2
Adjusting batch size: Rank 0: 64→56, Rank 2: 32→40
New cluster throughput: 475 samples/s
```

### Checkpoints

Models are saved automatically:
```
experiments/EXPERIMENT_NAME/checkpoints/
├── model_epoch_1.pth
├── model_epoch_5.pth
└── model_epoch_10.pth
```

### Completion

```
Training completed!
Total time: 1234.5 seconds
Final accuracy: 85.3%
Checkpoints saved to: experiments/EXPERIMENT_NAME/checkpoints/
```

---

## Troubleshooting During Training

### Issue: One worker not connecting

**Check worker status:**
```bash
docker ps  # On worker machine
docker logs hetero-worker-1
```

**Common causes:**
- Wrong MASTER_ADDR
- Firewall blocking port 29500
- Worker started too late (>60s)

**Solution:**
```bash
# Restart worker with correct settings
docker stop hetero-worker-1
docker rm hetero-worker-1

# Re-check environment variables
echo $MASTER_ADDR
echo $RANK
echo $WORLD_SIZE

# Restart
./scripts/docker_run_worker_adaptive.sh
```

### Issue: Training hangs at initialization

**Symptoms:** Stuck at "Initializing process group..."

**Causes:**
- WORLD_SIZE mismatch between nodes
- Not all workers started
- Network communication blocked

**Solution:**
```bash
# Verify WORLD_SIZE on all machines
# Master:
echo $WORLD_SIZE  # Should be same everywhere

# Workers:
echo $WORLD_SIZE  # Should match master

# Check all workers are running
docker ps  # On each worker machine
```

### Issue: Out of memory on one rank

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
```bash
# Reduce batch size for that specific rank
# Edit training script or use:
--batch-size 32  # Instead of 64
```

### Issue: Slow training on CPU workers

**This is normal!** CPU workers are 10-20x slower than GPU.

**Options:**
- Use smaller batch sizes on CPU (automatic in adaptive mode)
- Use smaller model for CPU workers
- Accept slower contribution (they still help!)

---

## Stopping and Restarting Training

### Graceful Shutdown

**On all machines:**
```bash
# Stop containers
docker stop hetero-master hetero-worker-1

# Remove containers
docker rm hetero-master hetero-worker-1
```

### Resuming Training

**Currently:** Training restarts from beginning.

**To add checkpointing:**
Edit training script to load from last checkpoint:
```python
--resume-from experiments/EXPERIMENT_NAME/checkpoints/model_epoch_5.pth
```

---

## Performance Tips

### 1. Use Same Docker Image Everywhere

✅ Pull from Docker Hub: Fast, consistent
❌ Build separately: Slower, version mismatch possible

### 2. Enable Load Balancing

```bash
--enable-load-balancing \
--load-balance-policy dynamic
```
Can improve throughput by 10-30% on heterogeneous clusters.

### 3. Use NCCL for All-GPU Clusters

```bash
--backend nccl
```
2-3x faster communication than Gloo.

### 4. Monitor GPU Utilization

```bash
# On GPU machines
watch -n 1 nvidia-smi
```
Aim for >80% GPU utilization.

### 5. Batch Size Tuning

- GPU: Start with 64, increase if memory allows
- CPU: Use 32 or lower
- Monitor GPU memory: Should be 70-90% used

---

## Example Full Training Command

For advanced users wanting full control:

```bash
docker run --rm \
  --name hetero-worker-1 \
  --network host \
  --gpus all \
  -v /path/to/project:/workspace \
  -v /path/to/data:/data \
  -e MASTER_ADDR=192.168.1.100 \
  -e MASTER_PORT=29500 \
  -e RANK=1 \
  -e WORLD_SIZE=4 \
  YOUR_USERNAME/hetero-cluster-training:latest \
  python -m src.training.main \
    --model resnet18 \
    --dataset cifar10 \
    --batch-size 64 \
    --epochs 50 \
    --learning-rate 0.1 \
    --enable-profiling \
    --enable-load-balancing \
    --load-balance-policy dynamic \
    --backend gloo \
    --master-addr 192.168.1.100 \
    --master-port 29500 \
    --experiment-name my_experiment
```

---

## Success Checklist

✅ Master started and waiting for workers
✅ All workers connected within 60 seconds
✅ No connection errors in logs
✅ Training started on all ranks
✅ Loss decreasing each epoch
✅ No out-of-memory errors
✅ Dashboard showing metrics (optional)
✅ Checkpoints being saved

---

## Results and Analysis

After training completes:

```bash
# View results
ls experiments/EXPERIMENT_NAME/

# Analyze performance
python scripts/analyze_results.py \
  --input-dir experiments/EXPERIMENT_NAME

# Compare experiments
streamlit run src/monitoring/dashboard.py
```

---

## Next Steps

1. ✅ Training running successfully
2. 📊 Monitor progress via dashboard
3. 📈 Analyze results after completion
4. 🔬 Experiment with different:
   - Models (simple_cnn, resnet18, resnet50)
   - Datasets (synthetic, cifar10, cifar100)
   - Load balancing policies
   - Batch sizes

---

## Quick Reference

**Master start:**
```bash
export WORLD_SIZE=4
./scripts/docker_run_master_adaptive.sh
```

**Worker start:**
```bash
export MASTER_ADDR=192.168.1.100
export RANK=1
export WORLD_SIZE=4
./scripts/docker_run_worker_adaptive.sh
```

**Monitor:**
```bash
docker logs -f hetero-master
./scripts/docker_run_dashboard.sh
```

**Stop:**
```bash
docker stop hetero-master hetero-worker-1
```

Happy distributed training! 🚀🎯
