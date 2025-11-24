# 🔧 Fix: Docker GPU Support

## Problem

You're seeing this error:
```
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]
```

This means Docker can't access your NVIDIA GPU.

---

## ✅ **Solution 1: Fix GPU Support (Recommended)**

### **Step 1: Install NVIDIA Container Toolkit**

Run the automated fix script:

```bash
sudo ./fix_docker_gpu.sh
```

This script will:
- ✅ Detect your GPU
- ✅ Install NVIDIA Container Toolkit
- ✅ Configure Docker
- ✅ Restart Docker daemon
- ✅ Test GPU access

### **Step 2: Verify**

After the script completes, test GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

You should see your GPU info!

### **Step 3: Run Your Setup**

Now you can run the master setup with GPU:

```bash
export WORLD_SIZE=2
./scripts/docker_run_master_adaptive.sh
```

---

## ⚡ **Solution 2: Use CPU Mode (Quick Test)**

If you just want to test the setup without GPU:

### **Option A: Use CPU Test Script**

```bash
./test_setup_cpu.sh
```

This builds a CPU-only Docker image and runs training.

### **Option B: Use Dockerfile.cpu**

```bash
# Build CPU image
docker build -f Dockerfile.cpu -t hetero-cluster-training-cpu .

# Run master in CPU mode
docker run --rm \
  --name hetero-master-cpu \
  --network host \
  -v "$(pwd):/workspace" \
  -e WORLD_SIZE=2 \
  hetero-cluster-training-cpu \
  python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --batch-size 32 \
    --epochs 2
```

---

## 🔍 **Manual Fix (If Script Fails)**

### **For Arch Linux:**

```bash
# Install NVIDIA Container Toolkit
sudo pacman -S nvidia-container-toolkit

# Or using AUR
yay -S nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker

# Test
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi
```

### **For Ubuntu/Debian:**

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

# Install
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

### **For Fedora/RHEL:**

```bash
# Add repository
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.repo | \
  sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo

# Install
sudo yum install -y nvidia-container-toolkit

# Configure Docker
sudo nvidia-ctk runtime configure --runtime=docker

# Restart Docker
sudo systemctl restart docker
```

---

## 🧪 **Testing**

After installation, verify GPU access:

```bash
# Test 1: Check GPU in Docker
docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu20.04 nvidia-smi

# Test 2: Run simple PyTorch test
docker run --rm --gpus all pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

Expected output:
```
CUDA available: True
GPU: NVIDIA GeForce RTX 3070
```

---

## ❓ **Troubleshooting**

### Issue: "nvidia-ctk: command not found"

**Solution:** The container toolkit didn't install properly. Try manual installation for your distro above.

### Issue: Still can't access GPU after installation

**Solution 1:** Reboot your system
```bash
sudo reboot
```

**Solution 2:** Check Docker daemon configuration
```bash
cat /etc/docker/daemon.json
```

Should contain:
```json
{
    "runtimes": {
        "nvidia": {
            "path": "nvidia-container-runtime",
            "runtimeArgs": []
        }
    }
}
```

**Solution 3:** Check Docker service
```bash
sudo systemctl status docker
sudo journalctl -u docker -n 50
```

### Issue: "Failed to initialize NVML"

**Solution:** GPU drivers issue. Reinstall NVIDIA drivers:
```bash
# Arch
sudo pacman -S nvidia nvidia-utils

# Ubuntu
sudo ubuntu-drivers autoinstall
```

Then reboot.

---

## 📝 **Why Does This Happen?**

Docker needs special runtime support to access NVIDIA GPUs. The regular Docker installation doesn't include this by default.

The **NVIDIA Container Toolkit** provides:
- GPU runtime for Docker
- CUDA libraries in containers
- GPU resource management
- Multi-GPU support

---

## 🎯 **Quick Decision Guide**

**Want GPU acceleration?**
→ Run `sudo ./fix_docker_gpu.sh`

**Just want to test the setup?**
→ Run `./test_setup_cpu.sh`

**Have multiple machines without GPUs?**
→ CPU mode works great for heterogeneous setups!

---

## ✅ **Success Indicators**

You've successfully fixed it when:

1. ✅ `nvidia-smi` works on host
2. ✅ `docker run --gpus all nvidia/cuda:... nvidia-smi` works
3. ✅ Your training script starts without GPU errors
4. ✅ GPU utilization shows in `nvidia-smi` while training

---

## 📚 **More Info**

- [NVIDIA Container Toolkit Docs](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- [Docker GPU Support](https://docs.docker.com/config/containers/resource_constraints/#gpu)
- [Arch Wiki: Docker](https://wiki.archlinux.org/title/Docker)
