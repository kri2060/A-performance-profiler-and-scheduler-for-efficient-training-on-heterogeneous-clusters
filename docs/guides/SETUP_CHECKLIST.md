# Multi-Worker Setup Checklist

Use this checklist for each machine in your cluster.

## Machine Information

- **Machine Name**: ________________
- **Role**: [ ] Master (RANK=0)  [ ] Worker (RANK=___)
- **IP Address**: ________________
- **OS**: [ ] Linux  [ ] Windows  [ ] macOS
- **Hardware**: [ ] GPU  [ ] CPU only

---

## Prerequisites Checklist

### For ALL Machines:

- [ ] **Docker installed**
  ```bash
  # Check Docker version
  docker --version
  ```
  - Expected: Docker version 20.10+

- [ ] **Network connectivity**
  ```bash
  # From worker, ping master IP
  ping 192.168.1.100
  ```
  - All machines must be on same network or VPN

- [ ] **Port 29500 accessible**
  ```bash
  # Linux: Check firewall
  sudo ufw status
  sudo ufw allow 29500

  # Windows: Add firewall rule
  # Open Windows Defender Firewall → New Inbound Rule → Port 29500
  ```

- [ ] **Project files available**
  - [ ] Either: Project copied to machine
  - [ ] Or: Docker image pulled from registry

### For GPU Machines Only:

- [ ] **NVIDIA GPU present**
  ```bash
  # Check GPU
  nvidia-smi
  ```
  - Should show GPU info without errors

- [ ] **NVIDIA Driver installed** (>= 450.80.02)
  ```bash
  # Check driver version
  nvidia-smi | grep "Driver Version"
  ```

- [ ] **NVIDIA Container Toolkit installed**
  ```bash
  # Linux: Install toolkit
  distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
  sudo apt-get update
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker

  # Verify
  docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
  ```

### For CPU-Only Machines:

- [ ] **Just Docker needed!** No GPU drivers required
  - CPU machines can still contribute to training

---

## Network Configuration

### Master Node:

1. [ ] **Find master IP address**
   ```bash
   # Linux/macOS
   hostname -I | awk '{print $1}'

   # Windows (PowerShell)
   (Get-NetIPAddress -AddressFamily IPv4 | Where-Object {$_.IPAddress -like "192.168.*"}).IPAddress
   ```

2. [ ] **Note your IP**: ________________
   - Share this IP with all worker machines

3. [ ] **Set WORLD_SIZE**
   - Count total machines (master + all workers)
   - Example: 1 master + 3 workers = WORLD_SIZE=4

### Worker Nodes:

1. [ ] **Get master IP from master node**
   - Master IP: ________________

2. [ ] **Assign unique RANK**
   - Worker 1: RANK=1
   - Worker 2: RANK=2
   - Worker 3: RANK=3
   - etc.

---

## Deployment Method Selection

Choose ONE deployment method:

### [ ] Option A: Docker Hub (Recommended - Fastest)
- **Pros**: No building, just pull image
- **Cons**: Requires Docker Hub account
- **Time**: ~2 minutes per machine

### [ ] Option B: Build Locally
- **Pros**: No registry needed
- **Cons**: Must build on each machine (~5 min)
- **Time**: ~5 minutes per machine

### [ ] Option C: Save/Load Image
- **Pros**: No internet needed, build once
- **Cons**: Must transfer large file (~3GB)
- **Time**: ~3 minutes + transfer time

---

## Ready to Launch?

Before starting training, verify:

- [ ] All machines have passed prerequisites
- [ ] All machines can reach master IP (ping test)
- [ ] Port 29500 is open on all machines
- [ ] Each worker has unique RANK assigned
- [ ] WORLD_SIZE is same on all machines
- [ ] You know which deployment option to use

---

## Quick Test Commands

### Test Docker:
```bash
docker run hello-world
```

### Test GPU in Docker (GPU machines):
```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### Test Network:
```bash
# From worker, ping master
ping <MASTER_IP>

# Check port (after master starts)
telnet <MASTER_IP> 29500
```

---

## Troubleshooting

### Docker not found
```bash
# Install Docker
# Linux (Ubuntu/Debian):
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
# Log out and back in

# Windows: Download from docker.com
```

### nvidia-smi not found
```bash
# Install NVIDIA drivers
# Ubuntu:
sudo ubuntu-drivers autoinstall
sudo reboot

# Check:
nvidia-smi
```

### Can't ping master
- Check if machines are on same network
- Check firewall settings
- Try disabling firewall temporarily for testing
- Use VPN if machines are on different networks

### Port 29500 already in use
```bash
# Use different port
export MASTER_PORT=29501
# Update on ALL machines
```

---

## Next Steps

Once checklist is complete, proceed to:
1. **Master Setup** → See MASTER_SETUP.md
2. **Worker Setup** → See WORKER_SETUP.md
3. **Launch Training** → See LAUNCH_GUIDE.md
