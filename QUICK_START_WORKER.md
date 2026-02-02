# Quick Start - Worker Setup (5 Minutes)

## On Worker Machine

### 1. Copy Project
```bash
# Transfer from master or clone repo
cd ~
git clone <repo-url>
cd A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters
```

### 2. Install PyTorch with CUDA 11.8
```bash
./setup_worker_cuda.sh
```

### 3. Get Master IP
Ask master for their IP, or check on master:
```bash
# On master
hostname -I
```

### 4. Configure Worker
```bash
nano START_WORKER.sh
```

Change these lines:
```bash
export MASTER_ADDR=10.161.199.68  # ← Master's IP
export WORLD_SIZE=3               # ← Must match master
export RANK=1                     # ← 1 for first worker, 2 for second, etc.
```

### 5. Fix Network (If Needed)

**Test connectivity:**
```bash
ping <master-ip>
nc -zv <master-ip> 29500
```

**If fails, on master run:**
```bash
sudo ufw allow from <worker-ip> to any port 29500 proto tcp
```

### 6. Start Training

**Start master first** (on master):
```bash
./master.sh
```

**Then start worker** (on worker):
```bash
./START_WORKER.sh
```

---

## Verification

Worker should show:
```
✓ NVIDIA GPU detected
Connecting to master at: 10.161.199.68:29500
Initialized process group: rank=1, world_size=3, backend=nccl
```

Master should show:
```
Waiting for 2 workers to connect...
All workers connected!
Starting training...
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "No route to host" | Fix firewall on master: `sudo ufw allow from <worker-ip> to any port 29500 proto tcp` |
| "ProcessGroupNCCL not available" | Run `./setup_worker_cuda.sh` |
| Worker hangs | Ensure master is running first |
| "WORLD_SIZE mismatch" | Must match on master and all workers |

---

## Files Created

- ✅ `setup_worker_cuda.sh` - Auto-install PyTorch with CUDA 11.8
- ✅ `WORKER_SETUP_GUIDE.md` - Detailed setup guide
- ✅ `QUICK_START_WORKER.md` - This quick reference

Transfer these files to worker machines!
