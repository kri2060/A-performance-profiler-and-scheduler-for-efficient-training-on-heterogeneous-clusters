# Enable All Features: Profiling, Load Balancing, Monitoring & Dashboard

**The COMPLETE guide to use ALL features of the Heterogeneous Cluster Trainer**

This guide ensures you get:
✅ Performance profiling (GPU usage, throughput, bottlenecks)
✅ Adaptive load balancing (automatic batch size adjustment)
✅ Real-time monitoring dashboard
✅ Heterogeneous scheduling
✅ Metrics for both master and workers

---

## What Went Wrong Before?

When you ran training, you used **basic PyTorch DDP** without the project's advanced features:

### What You Had:
```bash
python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --backend nccl
```

**Result:**
- ❌ No profiling metrics collected
- ❌ No adaptive load balancing
- ❌ Equal batch sizes (not optimized for heterogeneous GPUs)
- ❌ Empty dashboard
- ❌ No worker metrics visible

**This was just basic DDP!**

### What You Need:
```bash
python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --backend nccl \
  --enable-profiling \              # ✅ Collect performance metrics
  --enable-load-balancing \         # ✅ Adaptive batch sizing
  --load-balance-policy dynamic \   # ✅ Dynamic rebalancing
  --rebalance-interval 10 \         # ✅ Rebalance every 10 batches
  --experiment-name my_experiment   # ✅ Save organized metrics
```

---

## Complete Training Command with ALL Features

### Master (Linux)

```bash
cd "/home/kri2060/final project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters"

# Option 1: Native Python (recommended with WSL2)
source venv/bin/activate

export RANK=0
export WORLD_SIZE=2
export MASTER_ADDR=10.100.52.69
export MASTER_PORT=29500
export LOCAL_RANK=0

python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --lr 0.1 \
  --backend nccl \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --rebalance-interval 10 \
  --profile-interval 1 \
  --experiment-name resnet_hetero_training \
  --output-dir experiments

# Option 2: Docker
docker run --rm -it \
  --gpus all \
  --network host \
  --name hetero-master \
  -v "$(pwd):/workspace" \
  -e RANK=0 \
  -e WORLD_SIZE=2 \
  -e MASTER_ADDR=10.100.52.69 \
  -e MASTER_PORT=29500 \
  -e LOCAL_RANK=0 \
  kri2060/hetero-cluster-training \
  python -m src.training.main \
    --model resnet50 \
    --dataset cifar10 \
    --batch-size 128 \
    --epochs 100 \
    --lr 0.1 \
    --backend nccl \
    --master-addr 10.100.52.69 \
    --master-port 29500 \
    --enable-profiling \
    --enable-load-balancing \
    --load-balance-policy dynamic \
    --rebalance-interval 10 \
    --profile-interval 1 \
    --experiment-name resnet_hetero_training \
    --output-dir experiments
```

### Worker (Windows WSL2 or Linux)

```bash
cd ~/hetero-training
source venv/bin/activate

export RANK=1
export WORLD_SIZE=2
export MASTER_ADDR=10.100.52.69
export MASTER_PORT=29500
export LOCAL_RANK=0

python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --lr 0.1 \
  --backend nccl \
  --master-addr $MASTER_ADDR \
  --master-port $MASTER_PORT \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --rebalance-interval 10 \
  --profile-interval 1 \
  --experiment-name resnet_hetero_training \
  --output-dir experiments
```

---

## What Each Flag Does

### Core Training Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--model` | `resnet50` | Model architecture to train |
| `--dataset` | `cifar10` | Dataset to use (60K real images) |
| `--batch-size` | `128` | Base batch size (will be adjusted per GPU) |
| `--epochs` | `100` | Number of training epochs |
| `--lr` | `0.1` | Learning rate |
| `--backend` | `nccl` | Communication backend (nccl=fast, gloo=slow) |

### Profiling Flags (REQUIRED for metrics)

| Flag | Value | Purpose |
|------|-------|---------|
| `--enable-profiling` | *(no value)* | **Enable performance profiling** |
| `--profile-interval` | `1` | Profile every N iterations (1=every iter) |

**This enables:**
- ✅ GPU utilization tracking
- ✅ Memory usage tracking
- ✅ Throughput calculation (samples/sec)
- ✅ Iteration time breakdown (forward, backward, data loading)
- ✅ Bottleneck detection
- ✅ Metrics saved to JSON files

### Load Balancing Flags (REQUIRED for heterogeneous training)

| Flag | Value | Purpose |
|------|-------|---------|
| `--enable-load-balancing` | *(no value)* | **Enable adaptive load balancing** |
| `--load-balance-policy` | `dynamic` | Balancing strategy |
| `--rebalance-interval` | `10` | Rebalance every N batches |

**Load Balance Policies:**
- `proportional`: Adjust batch size based on GPU compute capability
- `dynamic`: Continuously adjust based on real-time performance
- `hybrid`: Combine both approaches

**This enables:**
- ✅ Automatic batch size adjustment per GPU
- ✅ Faster GPUs get larger batches
- ✅ Slower GPUs/CPUs get smaller batches
- ✅ Real-time performance monitoring
- ✅ Automatic rebalancing during training

### Organization Flags

| Flag | Value | Purpose |
|------|-------|---------|
| `--experiment-name` | `resnet_hetero_training` | Experiment name for organizing results |
| `--output-dir` | `experiments` | Root directory for all experiments |

**This creates:**
```
experiments/
└── resnet_hetero_training/
    ├── logs/
    │   ├── rank_0_metrics.json  ← Master metrics
    │   └── rank_1_metrics.json  ← Worker metrics
    └── configs/
        └── gpu_profiles.json     ← GPU hardware profiles
```

---

## What You'll See When It Works

### 1. Profiling Messages

```
INFO:profiling.performance_profiler:NVML enabled for GPU 0
INFO:profiling.performance_profiler:GPU 0: NVIDIA GeForce RTX 3050
INFO:profiling.performance_profiler:Memory: 4096 MB
```

### 2. Load Balancing Messages

```
INFO:scheduling.load_balancer:Registered 2 nodes
INFO:scheduling.load_balancer:Node 0: Compute score 8.6, Memory 4096MB
INFO:scheduling.load_balancer:Node 1: Compute score 8.6, Memory 4096MB
INFO:scheduling.load_balancer:Calculated batch sizes: {0: 128, 1: 128}
INFO:scheduling.load_balancer:Rebalancing triggered - performance variance detected
INFO:scheduling.load_balancer:New batch sizes: {0: 140, 1: 116}
```

### 3. Training Progress with Metrics

```
Rank 0 | Epoch 0 | Batch 10/195 | Loss: 2.245
Rank 1 | Epoch 0 | Batch 10/195 | Loss: 2.256

[Load Balancer Status]
Node 0: GPU 95%, Memory 65%, Throughput: 2450 samples/s
Node 1: GPU 92%, Memory 68%, Throughput: 2180 samples/s
```

### 4. Metrics Files Created

```bash
$ ls experiments/resnet_hetero_training/logs/
rank_0_metrics.json
rank_1_metrics.json

$ head experiments/resnet_hetero_training/logs/rank_0_metrics.json
[
  {
    "epoch": 0,
    "iteration": 0,
    "loss": 2.3045,
    "throughput": 2450.5,
    "iteration_time": 0.052,
    "data_loading_time": 0.008,
    "forward_time": 0.018,
    "backward_time": 0.022,
    "optimizer_time": 0.004,
    "gpu_utilization": 95.2,
    "gpu_memory_percent": 64.8
  },
  ...
]
```

### 5. Dashboard Shows Real Data

**Current Statistics:**
```
Rank 0 - GPU Util: 95.2%
Rank 0 - Memory: 64.8%
Rank 0 - Throughput: 2450 samples/s

Rank 1 - GPU Util: 92.1%
Rank 1 - Memory: 68.2%
Rank 1 - Throughput: 2180 samples/s
```

**With live graphs showing both workers!**

---

## Start Dashboard to Monitor

### Option 1: Native Python

```bash
cd "/home/kri2060/final project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters"

source venv/bin/activate

# Install Streamlit if not already
pip install streamlit

# Start dashboard
streamlit run src/monitoring/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

### Option 2: Docker

```bash
docker run --rm -d \
  --name hetero-dashboard \
  -p 8501:8501 \
  -v "$(pwd):/workspace" \
  kri2060/hetero-cluster-training \
  streamlit run src/monitoring/dashboard.py --server.port=8501 --server.address=0.0.0.0
```

**Access:** http://10.100.52.69:8501

---

## Step-by-Step Full Setup

### Step 1: Prepare Directories

```bash
cd "/home/kri2060/final project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters"

# Clean old experiments (optional)
rm -rf experiments/distributed_training

# Create fresh experiment directory
mkdir -p experiments/resnet_hetero_training/{logs,configs}
```

### Step 2: Start Master (with ALL flags)

```bash
source venv/bin/activate

export RANK=0 WORLD_SIZE=2 MASTER_ADDR=10.100.52.69 MASTER_PORT=29500 LOCAL_RANK=0

python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --lr 0.1 \
  --backend nccl \
  --master-addr $MASTER_ADDR \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --experiment-name resnet_hetero_training
```

### Step 3: Start Worker (WSL2)

```bash
cd ~/hetero-training
source venv/bin/activate

export RANK=1 WORLD_SIZE=2 MASTER_ADDR=10.100.52.69 MASTER_PORT=29500 LOCAL_RANK=0

python -m src.training.main \
  --model resnet50 \
  --dataset cifar10 \
  --batch-size 128 \
  --epochs 100 \
  --lr 0.1 \
  --backend nccl \
  --master-addr $MASTER_ADDR \
  --enable-profiling \
  --enable-load-balancing \
  --load-balance-policy dynamic \
  --experiment-name resnet_hetero_training
```

### Step 4: Start Dashboard

**In another terminal on master:**

```bash
source venv/bin/activate
streamlit run src/monitoring/dashboard.py --server.port 8501 --server.address 0.0.0.0
```

**Open browser:** http://10.100.52.69:8501

---

## Verify Everything Works

### Check 1: Profiling Enabled

**Look for these log messages:**
```
✅ INFO:profiling.performance_profiler:NVML enabled for GPU 0
✅ INFO:profiling.performance_profiler:Starting iteration profiling
```

### Check 2: Load Balancing Active

**Look for these log messages:**
```
✅ INFO:scheduling.load_balancer:Registered 2 nodes
✅ INFO:scheduling.load_balancer:Calculated batch sizes
✅ INFO:scheduling.load_balancer:Node 0: batch_size=128
```

### Check 3: Metrics Being Saved

```bash
# Check metrics files exist and are growing
watch -n 1 'ls -lh experiments/resnet_hetero_training/logs/'

# Should show:
# rank_0_metrics.json (growing in size)
# rank_1_metrics.json (growing in size)
```

### Check 4: Dashboard Shows Data

**Dashboard should display:**
- ✅ Current statistics for BOTH ranks
- ✅ GPU utilization graphs for both workers
- ✅ Training loss curves
- ✅ Throughput comparison
- ✅ Iteration time breakdown
- ✅ Bottleneck analysis

### Check 5: Load Balancing Triggers

**After 10-20 batches, you should see:**
```
[Load Balancer Status]
╔════════════════════════════════════════════╗
║ Adaptive Load Balancer Status              ║
╠════════════════════════════════════════════╣
║ Node 0: GPU 95% | Memory 65% | Batch: 140 ║
║ Node 1: GPU 92% | Memory 68% | Batch: 116 ║
╠════════════════════════════════════════════╣
║ Cluster Throughput: 4630 samples/s        ║
╚════════════════════════════════════════════╝
```

---

## Common Issues and Solutions

### Issue 1: "No metrics found" in Dashboard

**Cause:** Forgot `--enable-profiling` flag

**Solution:**
```bash
# Add these flags:
--enable-profiling \
--enable-load-balancing \
--experiment-name resnet_hetero_training
```

### Issue 2: Only Rank 0 shown in Dashboard

**Cause:** Worker metrics not being saved

**Check:**
```bash
# Both files should exist:
ls experiments/resnet_hetero_training/logs/
# rank_0_metrics.json  ← Master
# rank_1_metrics.json  ← Worker
```

**Solution:** Make sure worker also has `--enable-profiling --experiment-name resnet_hetero_training`

### Issue 3: Load Balancing Not Working

**Symptoms:**
- No batch size adjustments
- No load balancer status messages

**Check:**
```bash
# Look for this in logs:
grep "load_balancer" <terminal output>
```

**Solution:** Add all three flags:
```bash
--enable-load-balancing \
--load-balance-policy dynamic \
--rebalance-interval 10
```

### Issue 4: Different experiment names

**Problem:** Master and worker using different experiment names

**Solution:** Use SAME experiment name on both:
```bash
# Both must use:
--experiment-name resnet_hetero_training
```

---

## Quick Test to Verify Features Work

**Test with small dataset first:**

```bash
# Master
python -m src.training.main \
  --model simple_cnn \
  --dataset synthetic_image \
  --num-samples 1000 \
  --batch-size 32 \
  --epochs 2 \
  --backend nccl \
  --master-addr 10.100.52.69 \
  --enable-profiling \
  --enable-load-balancing \
  --experiment-name quick_test

# Worker
python -m src.training.main \
  --model simple_cnn \
  --dataset synthetic_image \
  --num-samples 1000 \
  --batch-size 32 \
  --epochs 2 \
  --backend nccl \
  --master-addr 10.100.52.69 \
  --enable-profiling \
  --enable-load-balancing \
  --experiment-name quick_test
```

**Takes 1 minute, then check:**
```bash
ls experiments/quick_test/logs/
# Should have rank_0_metrics.json and rank_1_metrics.json
```

---

## Summary Checklist

Before running training, ensure:

- [ ] Using `--enable-profiling` flag
- [ ] Using `--enable-load-balancing` flag
- [ ] Using `--experiment-name` flag (same on all workers!)
- [ ] Using `--backend nccl` (for best performance)
- [ ] Experiment directory created: `experiments/<name>/logs/`
- [ ] Dashboard started after training begins
- [ ] Both master and worker have SAME flags

**Then you'll get:**
- ✅ Real-time profiling metrics
- ✅ Adaptive load balancing
- ✅ Live dashboard with all workers
- ✅ Heterogeneous scheduling
- ✅ Performance bottleneck analysis

---

## Next Steps

1. **Run test** with small model to verify features work
2. **Start dashboard** to see live metrics
3. **Run full training** with ResNet50 + CIFAR10
4. **Monitor adaptive load balancing** in action
5. **Analyze results** from metrics JSON files

**Now you'll see the REAL power of heterogeneous cluster training!** 🚀
