# Documentation Guide Index

All setup and usage guides for the Heterogeneous Cluster Training project.

---

## 🚀 Quick Start

**New to the project? Start here:**

1. **[QUICK_START_MULTINODE.md](QUICK_START_MULTINODE.md)** - Get training in 5 minutes
2. **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)** - ⭐ **RECOMMENDED** for Windows + Linux (80-100% GPU usage)
3. **[ENABLE_ALL_FEATURES.md](ENABLE_ALL_FEATURES.md)** - 🔥 **IMPORTANT** Enable profiling, load balancing & monitoring

---

## 📚 Complete Guides

### Setup Guides

- **[SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)** - Prerequisites for all machines
- **[MASTER_SETUP.md](MASTER_SETUP.md)** - Master node setup (RANK=0)
- **[WORKER_SETUP.md](WORKER_SETUP.md)** - Worker node setup (RANK 1, 2, 3...)
- **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)** - WSL2 + NCCL for maximum performance (No Docker)
- **[ENABLE_ALL_FEATURES.md](ENABLE_ALL_FEATURES.md)** - Enable profiling, load balancing & dashboard

### Docker Guides

- **[DOCKER_MULTINODE_SETUP.md](DOCKER_MULTINODE_SETUP.md)** - Comprehensive Docker guide
- **[DOCKER_DEPLOYMENT_GUIDE.md](DOCKER_DEPLOYMENT_GUIDE.md)** - Docker deployment
- **[DOCKER_GPU_FIX.md](DOCKER_GPU_FIX.md)** - GPU issues in Docker

### Training & Usage

- **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** - Complete training examples & scenarios
- **[QUICKSTART.md](QUICKSTART.md)** - Single-machine quick start

### Reference

- **[MULTINODE_INDEX.md](MULTINODE_INDEX.md)** - Documentation index
- **[SETUP_DIAGRAM.md](SETUP_DIAGRAM.md)** - Architecture diagrams
- **[HOW_TO_USE_DOCS_WEBSITE.md](HOW_TO_USE_DOCS_WEBSITE.md)** - Documentation website

---

## 🎯 Choose Your Path

### Path 1: Maximum Performance (Recommended)

**Use WSL2 + NCCL (No Docker)**

→ **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)**

**Benefits:**
- ✅ 80-100% GPU utilization
- ✅ 5x faster training
- ✅ Simpler setup (no Docker)
- ✅ Direct GPU access

**Best for:**
- Windows + Linux mixed cluster
- Want maximum performance
- Have NVIDIA GPUs

---

### Path 2: Quick Testing (Docker)

**Use Docker with Gloo backend**

→ **[QUICK_START_MULTINODE.md](QUICK_START_MULTINODE.md)**

**Benefits:**
- ✅ Works on any OS
- ✅ Isolated environments
- ✅ Easy deployment

**Limitations:**
- ⚠️ Lower GPU usage (1-12%)
- ⚠️ Slower training
- ⚠️ Gloo backend overhead

**Best for:**
- Quick testing
- Verifying cluster setup
- Learning the system

---

### Path 3: Production Docker

**Use Docker Hub with custom configuration**

→ **[DOCKER_MULTINODE_SETUP.md](DOCKER_MULTINODE_SETUP.md)**

**Best for:**
- Production deployments
- Team environments
- Reproducible setups

---

## 📊 Performance Comparison

| Method | GPU Usage | Training Speed | Setup Complexity |
|--------|-----------|----------------|------------------|
| **WSL2 + NCCL** | 80-100% ⚡ | **Fastest** | Low |
| Docker + NCCL (Linux only) | 80-100% ⚡ | Fast | Medium |
| Docker + Gloo | 1-12% 🐌 | Slow | Low |

---

## 🔧 By Use Case

### I want maximum GPU utilization
→ **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)**

### I'm setting up for the first time
→ **[QUICK_START_MULTINODE.md](QUICK_START_MULTINODE.md)**

### I have mixed GPU/CPU hardware
→ **[LAUNCH_GUIDE.md](LAUNCH_GUIDE.md)** (Scenario 1)

### I have Windows + Linux machines
→ **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)**

### I need to troubleshoot connections
→ **[WORKER_SETUP.md](WORKER_SETUP.md)** (Troubleshooting section)

### I want to use Docker Hub
→ **[MASTER_SETUP.md](MASTER_SETUP.md)** (Method A)

---

## 📁 File Organization

```
docs/guides/
├── README.md                      ← You are here
├── WSL2_NCCL_SETUP.md            ← ⭐ Recommended setup
├── QUICK_START_MULTINODE.md      ← 5-minute quick start
├── SETUP_CHECKLIST.md            ← Prerequisites
├── MASTER_SETUP.md               ← Master node setup
├── WORKER_SETUP.md               ← Worker node setup
├── LAUNCH_GUIDE.md               ← Training examples
├── DOCKER_MULTINODE_SETUP.md     ← Comprehensive Docker
├── DOCKER_DEPLOYMENT_GUIDE.md    ← Docker deployment
├── DOCKER_GPU_FIX.md             ← GPU troubleshooting
├── MULTINODE_INDEX.md            ← Navigation index
├── QUICKSTART.md                 ← Single-machine start
├── SETUP_DIAGRAM.md              ← Architecture
└── HOW_TO_USE_DOCS_WEBSITE.md   ← Docs website
```

---

## 🆘 Getting Help

1. **Check the guides** - Most questions answered here
2. **Troubleshooting sections** - Each guide has troubleshooting
3. **Check logs** - `docker logs <container-name>`
4. **Verify setup** - Use [SETUP_CHECKLIST.md](SETUP_CHECKLIST.md)

---

## 🎓 Recommended Learning Path

### Beginner
1. [QUICKSTART.md](QUICKSTART.md) - Single machine
2. [QUICK_START_MULTINODE.md](QUICK_START_MULTINODE.md) - Multiple machines
3. [WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md) - Optimize performance

### Intermediate
1. [LAUNCH_GUIDE.md](LAUNCH_GUIDE.md) - Advanced training
2. [DOCKER_MULTINODE_SETUP.md](DOCKER_MULTINODE_SETUP.md) - Docker deep dive

### Advanced
1. Modify [src/training/main.py](../../src/training/main.py)
2. Implement custom load balancing
3. Add new models and datasets

---

## 🔥 Most Popular Setup (Dec 2024)

**WSL2 + NCCL without Docker**

Why? Because it gives you:
- ✅ Maximum GPU utilization (80-100%)
- ✅ Simplest setup
- ✅ Fastest training
- ✅ Works with Windows + Linux

→ **[WSL2_NCCL_SETUP.md](WSL2_NCCL_SETUP.md)**

---

**Ready to start? Pick a guide above and let's train!** 🚀
