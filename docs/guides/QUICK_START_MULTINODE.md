# Multi-Node Quick Start (5 Minutes)

Get your cluster training in 5 minutes! Follow this streamlined guide.

---

## 🎯 What You'll Do

1. Setup master node (2 min)
2. Setup worker nodes (1 min each)
3. Start training (automatic)

---

## 📋 Before You Start

You need:
- [ ] 2+ machines on same network
- [ ] Docker installed on each machine
- [ ] Master node IP address
- [ ] (Optional) NVIDIA GPU + drivers

---

## 🚀 Step 1: Master Node (2 minutes)

**On your main laptop/server:**

```bash
cd /home/kri2060/final\ project/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Set total number of machines
export WORLD_SIZE=4  # 1 master + 3 workers

# Start master
chmod +x scripts/docker_run_master_adaptive.sh
./scripts/docker_run_master_adaptive.sh
```

**You'll see:**
```
✓ Master IP: 192.168.1.100
⏳ Waiting 60 seconds for workers...
```

**→ Share this IP with your team: `192.168.1.100`**

---

## 🔧 Step 2: Worker Nodes (1 min each)

**On each worker laptop:**

### Linux/Mac Workers:

```bash
# Get project
cd ~/project  # Or clone/copy project first

# Configure
export MASTER_ADDR=192.168.1.100  # From master
export RANK=1                      # Unique: 1, 2, 3...
export WORLD_SIZE=4                # Same as master

# Start worker
chmod +x scripts/docker_run_worker_adaptive.sh
./scripts/docker_run_worker_adaptive.sh
```

### Windows Workers:

```cmd
REM Open PowerShell as Admin
cd C:\project

REM Configure
set MASTER_ADDR=192.168.1.100
set RANK=1
set WORLD_SIZE=4

REM Start worker
scripts\docker_run_worker_adaptive.bat
```

**⚠️ Important:** Each worker needs unique RANK (1, 2, 3, ...)

---

## ✅ Step 3: Training Starts Automatically!

Once all workers connect, training begins automatically.

**Monitor progress:**
```bash
# Watch logs
docker logs -f hetero-master

# Or start dashboard
./scripts/docker_run_dashboard.sh
# Open: http://localhost:8501
```

---

## 🎉 That's It!

Your cluster is training. You'll see:

```
Epoch 1/10:
  Rank 0: 256 samples/s (GPU)
  Rank 1: 180 samples/s (GPU)
  Rank 2: 45 samples/s (CPU)
  Rank 3: 40 samples/s (CPU)

Cluster throughput: 521 samples/s
```

---

## 🐛 Troubleshooting (Most Common Issues)

### Workers can't connect?

**Check firewall:**
```bash
# Linux
sudo ufw allow 29500

# Windows: Open port 29500 in Windows Defender Firewall
```

**Test connection:**
```bash
ping 192.168.1.100
```

### Docker not found?

**Install Docker:**
```bash
# Linux
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Windows: Download from docker.com
```

### Wrong RANK?

Each worker needs **unique RANK**:
- Worker 1: `RANK=1`
- Worker 2: `RANK=2`
- Worker 3: `RANK=3`

---

## 📚 Detailed Guides

For more details, see:

1. **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Prerequisites
2. **[MASTER_SETUP.md](MASTER_SETUP.md)** - Detailed master setup
3. **[WORKER_SETUP.md](WORKER_SETUP.md)** - Detailed worker setup
4. **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - Complete examples
5. **[DOCKER_MULTINODE_SETUP.md](DOCKER_MULTINODE_SETUP.md)** - Full Docker guide

---

## ⚡ Using Docker Hub (Even Faster)

Skip building on each machine:

### Master (one-time):

```bash
# Login to Docker Hub
docker login

# Build and push
docker build -t YOUR_USERNAME/hetero-cluster:latest .
docker push YOUR_USERNAME/hetero-cluster:latest
```

### All Workers:

```bash
# Just pull and run
docker pull YOUR_USERNAME/hetero-cluster:latest
export IMAGE_NAME=YOUR_USERNAME/hetero-cluster:latest
./scripts/docker_run_worker_adaptive.sh
```

**→ Saves ~5 minutes per worker!**

---

## 🎓 Common Setups

### 4 Machines (2 GPU + 2 CPU):

```
Master (RANK=0): Linux + RTX 3070 @ 192.168.1.100
Worker 1 (RANK=1): Windows + GTX 1660 @ 192.168.1.101
Worker 2 (RANK=2): Linux CPU @ 192.168.1.102
Worker 3 (RANK=3): Windows CPU @ 192.168.1.103

WORLD_SIZE=4
```

### 3 All-GPU:

```
Master (RANK=0): GPU @ 192.168.1.100
Worker 1 (RANK=1): GPU @ 192.168.1.101
Worker 2 (RANK=2): GPU @ 192.168.1.102

WORLD_SIZE=3
Use: --backend nccl (faster)
```

### 2 CPU-Only:

```
Master (RANK=0): CPU @ 192.168.1.100
Worker 1 (RANK=1): CPU @ 192.168.1.101

WORLD_SIZE=2
Perfect for testing!
```

---

## 📊 Performance Tips

**Enable adaptive load balancing:**

Edit `scripts/docker_run_master_adaptive.sh` and add:
```bash
--enable-load-balancing \
--load-balance-policy dynamic
```

**Use NCCL for all-GPU clusters:**
```bash
--backend nccl  # Instead of gloo
```

**Tune batch sizes:**
- GPU: 64-128
- CPU: 32

---

## 🛑 Stopping Training

```bash
# On all machines
docker stop hetero-master hetero-worker-1
docker rm hetero-master hetero-worker-1
```

---

## 📈 View Results

```bash
# Check results
ls experiments/multi_node_training/

# Start dashboard
streamlit run src/monitoring/dashboard.py
```

---

## 🔄 Environment Variables Cheat Sheet

### Master:
```bash
export WORLD_SIZE=4
export EXPERIMENT_NAME="my_experiment"
```

### Workers:
```bash
export MASTER_ADDR=192.168.1.100  # From master
export MASTER_PORT=29500           # Default
export RANK=1                      # UNIQUE!
export WORLD_SIZE=4                # Same as master
export EXPERIMENT_NAME="my_experiment"
```

---

## ✨ Advanced: Full Custom Command

```bash
docker run --rm --name hetero-worker-1 --network host \
  -e MASTER_ADDR=192.168.1.100 \
  -e RANK=1 \
  -e WORLD_SIZE=4 \
  hetero-cluster-training:latest \
  python -m src.training.main \
    --model resnet18 \
    --dataset cifar10 \
    --batch-size 64 \
    --epochs 50 \
    --enable-load-balancing \
    --master-addr 192.168.1.100
```

---

## 💡 Tips

1. **Start master first**, then workers within 60 seconds
2. **Each worker needs unique RANK** (1, 2, 3...)
3. **Same WORLD_SIZE on all machines**
4. **GPU/CPU auto-detected** - no configuration needed
5. **Port 29500** must be open on master

---

## 🎯 Success Checklist

- [ ] Master started and shows IP
- [ ] All workers connected (check `docker ps`)
- [ ] Training started (check `docker logs hetero-master`)
- [ ] No errors in logs
- [ ] Loss decreasing each epoch

---

## 📞 Need Help?

1. Check [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) for detailed examples
2. See troubleshooting in [DOCKER_MULTINODE_SETUP.md](DOCKER_MULTINODE_SETUP.md)
3. Review logs: `docker logs hetero-master`

---

**Ready to scale your training! 🚀**

Got 4 laptops? Get 4x the speed! (approximately)

The system automatically:
- ✅ Detects GPU/CPU
- ✅ Adjusts batch sizes
- ✅ Balances workload
- ✅ Handles heterogeneous hardware

Just start the nodes and let it train! 🎉
