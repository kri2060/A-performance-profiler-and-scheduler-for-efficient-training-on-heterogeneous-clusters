# Worker Node Setup Guide

Complete guide to set up a worker node for distributed training.

## Prerequisites

- Worker machine with NVIDIA GPU
- Same network as master node
- Python 3.9+ installed
- Git installed

---

## Step 1: Copy Project to Worker

On the **worker machine**:

```bash
# Clone the repository
cd ~
git clone <your-repo-url>
cd A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Or copy from master using scp
# From master machine:
# scp -r /path/to/project worker@<worker-ip>:~/
```

---

## Step 2: Install NVIDIA Drivers (If Not Installed)

```bash
# Check if drivers are installed
nvidia-smi

# If not installed, install drivers
sudo apt update
sudo apt install nvidia-driver-545  # Or latest stable version

# Reboot after installation
sudo reboot

# Verify after reboot
nvidia-smi
```

---

## Step 3: Install Matching PyTorch with CUDA 11.8

Run the automated setup script:

```bash
cd ~/A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# Run setup script
./setup_worker_cuda.sh
```

This will:
- Create virtual environment
- Install PyTorch 2.6.0 with CUDA 11.8
- Install torchvision 0.19.0
- Verify CUDA functionality

**Manual Installation (Alternative):**

```bash
# Create and activate venv
python3 -m venv venv
source venv/bin/activate

# Install PyTorch with CUDA 11.8
pip install torch==2.6.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
pip install -r requirements.txt

# Verify
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

---

## Step 4: Configure Network Connectivity

### A. Get Master's IP Address

On **master node**, run:
```bash
./get_ip.sh
# Or manually:
ip addr show | grep "inet " | grep -v "127.0.0.1"
```

Note the master's IP (e.g., `10.161.199.68`)

### B. Test Connectivity from Worker

On **worker node**:
```bash
# Test ping
ping <master-ip>

# Test if port 29500 is reachable
nc -zv <master-ip> 29500
# Or use telnet
telnet <master-ip> 29500
```

### C. Fix Firewall on Master (If Connection Fails)

If you get "No route to host" or "Connection refused":

**On master node**, run:
```bash
# Get worker's IP
# On worker: hostname -I

# Allow worker to connect to master
sudo ufw allow from <worker-ip> to any port 29500 proto tcp

# Or allow entire subnet (easier for multiple workers)
sudo ufw allow from 10.161.199.0/24 to any port 29500 proto tcp

# Check firewall status
sudo ufw status verbose
```

**Common Issues:**
- **WiFi AP Isolation**: Check your router settings and disable "AP Isolation" or "Client Isolation"
- **Different networks**: Ensure both machines are on the same WiFi/Ethernet network
- **Firewall blocking**: Temporarily disable firewall for testing: `sudo ufw disable` (re-enable after: `sudo ufw enable`)

---

## Step 5: Update Worker Configuration

Edit `START_WORKER.sh` on the **worker node**:

```bash
nano START_WORKER.sh
```

Update these lines:
```bash
export RANK=1                        # First worker = 1, second = 2, etc.
export WORLD_SIZE=3                  # Must match master (total nodes)
export MASTER_ADDR=10.161.199.68    # ← CHANGE THIS: Your master's IP
export MASTER_PORT=29500             # Should match master
```

Save and exit (Ctrl+X, Y, Enter)

---

## Step 6: Start Worker Training

On **worker node**:

```bash
# Make script executable
chmod +x START_WORKER.sh

# Start worker
./START_WORKER.sh
```

You should see:
```
==========================================
Starting Worker Node (Rank 1) - Linux
==========================================
✓ NVIDIA GPU detected
CUDA device available:
NVIDIA GeForce GTX 1650, 4096 MiB
Using network interface: wlan0
Connecting to master at: 10.161.199.68:29500
==========================================
```

The worker will:
1. Wait for master to start
2. Connect to master's process group
3. Begin training synchronously

---

## Verification Checklist

### ✓ Step-by-Step Verification

- [ ] NVIDIA drivers installed (`nvidia-smi` works)
- [ ] PyTorch 2.6.0 with CUDA 11.8 installed
- [ ] Virtual environment activated
- [ ] Can ping master node
- [ ] Can connect to master port 29500 (`nc -zv <master-ip> 29500`)
- [ ] `START_WORKER.sh` configured with correct:
  - [ ] `MASTER_ADDR` (master's IP)
  - [ ] `WORLD_SIZE` (matches master)
  - [ ] `RANK` (unique for each worker: 1, 2, 3...)
- [ ] Master node is already running (`./master.sh`)
- [ ] Worker started successfully

---

## Common Issues and Solutions

### Issue 1: "No route to host"

**Symptom:**
```
[c10d] The client socket has failed to connect to [::ffff:10.161.199.68]:29500 (errno: 113 - No route to host)
```

**Solution:**
1. Check if both machines are on same network
2. Disable WiFi AP Isolation on router
3. Allow port 29500 on master's firewall:
   ```bash
   sudo ufw allow from <worker-ip> to any port 29500 proto tcp
   ```

### Issue 2: "ProcessGroupNCCL is only supported with GPUs"

**Symptom:**
```
WARNING: Failed to initialize with nccl, falling back to gloo
```

**Solution:**
- This is **NORMAL** if worker doesn't have GPU or CUDA isn't available
- The system automatically falls back to CPU (Gloo) backend
- If you have a GPU, verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- If False, reinstall PyTorch with CUDA 11.8: `./setup_worker_cuda.sh`

### Issue 3: "Connection timeout"

**Symptom:**
Worker hangs with no output after "Connecting to master..."

**Solution:**
1. Ensure master is running FIRST
2. Check `WORLD_SIZE` matches on both nodes
3. Verify network connectivity
4. Check firewall on BOTH master and worker

### Issue 4: "CUDA out of memory"

**Symptom:**
```
RuntimeError: CUDA out of memory
```

**Solution:**
- Reduce batch size in `START_WORKER.sh`:
  ```bash
  python -u -m src.training.main \
    --batch-size 16  # ← Reduce this (default is 32)
  ```
- Or enable dynamic batch sizing (already in code)

### Issue 5: "Module not found"

**Symptom:**
```
ModuleNotFoundError: No module named 'torch'
```

**Solution:**
```bash
source venv/bin/activate
pip install -r requirements.txt
./setup_worker_cuda.sh
```

---

## Multi-Worker Setup

To add a **second worker** (Rank 2):

1. Follow all steps above on the second worker machine
2. In `START_WORKER.sh`, change:
   ```bash
   export RANK=2  # ← Different rank for each worker
   ```
3. On **master**, update `master.sh`:
   ```bash
   export WORLD_SIZE=3  # Master + 2 workers = 3 total
   ```

---

## Testing Setup

### Quick Test Script

On **worker**, create `test_worker.py`:

```python
import torch
import torch.distributed as dist

# Initialize process group
dist.init_process_group(
    backend='gloo',  # or 'nccl' for GPU
    init_method='tcp://10.161.199.68:29500',  # Master IP
    world_size=2,
    rank=1
)

print(f"✓ Connected! Rank: {dist.get_rank()}, World size: {dist.get_world_size()}")

# Test tensor operation
tensor = torch.tensor([1.0, 2.0, 3.0])
if torch.cuda.is_available():
    tensor = tensor.cuda()
    print(f"✓ CUDA working: {tensor.device}")

dist.destroy_process_group()
print("✓ Test complete!")
```

Run:
```bash
source venv/bin/activate
python test_worker.py
```

---

## Performance Tips

1. **Use Ethernet instead of WiFi** for better bandwidth
2. **Disable power management** on GPUs:
   ```bash
   sudo nvidia-smi -pm 1
   ```
3. **Set GPU to performance mode**:
   ```bash
   sudo nvidia-smi -ac <mem_clock>,<gpu_clock>
   ```
4. **Monitor GPU during training**:
   ```bash
   watch -n 1 nvidia-smi
   ```

---

## Master Node Configuration

Master's `WORLD_SIZE` must equal **total number of nodes**:

| Setup | Master Rank | Worker Ranks | WORLD_SIZE |
|-------|------------|--------------|------------|
| 1 Master + 1 Worker | 0 | 1 | 2 |
| 1 Master + 2 Workers | 0 | 1, 2 | 3 |
| 1 Master + 3 Workers | 0 | 1, 2, 3 | 4 |

Each node (master + workers) must have:
- **Unique RANK** (0 for master, 1+ for workers)
- **Same WORLD_SIZE**
- **Same MASTER_ADDR and MASTER_PORT**

---

## Summary

**On Worker:**
1. ✅ Install NVIDIA drivers
2. ✅ Run `./setup_worker_cuda.sh`
3. ✅ Fix network connectivity
4. ✅ Update `START_WORKER.sh` with master IP
5. ✅ Start master first: `./master.sh` (on master)
6. ✅ Start worker: `./START_WORKER.sh` (on worker)

**Files to Transfer:**
- Entire project directory
- Or just: `requirements.txt`, `START_WORKER.sh`, `setup_worker_cuda.sh`, `src/` folder

**Key Settings:**
- Master IP: `<check with ./get_ip.sh on master>`
- Port: `29500`
- Backend: `nccl` (GPU) or `gloo` (CPU fallback)

Good luck! 🚀
