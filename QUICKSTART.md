# Quick Start Guide

Get started with heterogeneous cluster training in 5 minutes!

## Prerequisites

- Python 3.9+
- At least 1 GPU (preferably 2+ for heterogeneous testing)
- 10 GB disk space

## Installation (2 minutes)

```bash
# 1. Clone repository
git clone <your-repo-url>
cd A-performance-profiler-and-scheduler-for-efficient-training-on-heterogeneous-clusters

# 2. Install dependencies
pip install -r requirements.txt

# 3. Install package
pip install -e .
```

## First Run (3 minutes)

### Step 1: Profile Your GPUs

```bash
python -m src.profiling.main
```

Expected output:
```
Found 2 GPU(s)
Profiling GPU 0...
GPU 0 benchmark: 8.45 TFLOPS
GPU 0: NVIDIA GeForce RTX 3060 (Score: 8.5)
...
Saved GPU profiles to experiments/configs/gpu_profiles.json
```

### Step 2: Run Quick Test

```bash
python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --num-samples 1000 \
    --batch-size 32 \
    --epochs 2 \
    --enable-profiling \
    --experiment-name quick_test
```

### Step 3: View Results

```bash
streamlit run src/monitoring/dashboard.py
```

Open http://localhost:8501 in your browser.

## Test Load Balancing

Compare baseline vs. dynamic load balancing:

```bash
# Baseline (no load balancing)
python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --num-samples 5000 \
    --batch-size 32 \
    --epochs 5 \
    --enable-profiling \
    --experiment-name baseline

# With dynamic load balancing
python -m src.training.main \
    --model simple_cnn \
    --dataset synthetic_image \
    --num-samples 5000 \
    --batch-size 32 \
    --epochs 5 \
    --enable-profiling \
    --enable-load-balancing \
    --load-balance-policy dynamic \
    --experiment-name dynamic
```

Compare results in the dashboard!

## Full Benchmark

Run comprehensive benchmarks:

```bash
bash scripts/run_benchmark.sh
```

This will take 15-30 minutes depending on your hardware.

## Troubleshooting

### Issue: NCCL not available

**Solution**: Use Gloo backend instead
```bash
--backend gloo
```

### Issue: Out of memory

**Solution**: Reduce batch size
```bash
--batch-size 16
```

### Issue: No GPU found

**Solution**: Install CUDA and PyTorch with GPU support
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: NVML initialization failed

**Solution**: Install NVIDIA drivers
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-driver-525

# Verify
nvidia-smi
```

## Next Steps

- Read the full [README.md](README.md)
- Try different models: `resnet50`, `bert`, `gpt2`
- Experiment with different policies: `proportional`, `dynamic`, `hybrid`
- Check out example scripts in `scripts/`
- Run on real datasets: `cifar10`, `cifar100`

## Common Commands

```bash
# Profile hardware
python -m src.profiling.main

# Quick training test
python -m src.training.main --model simple_cnn --dataset synthetic_image --epochs 2

# Training with load balancing
python -m src.training.main --enable-load-balancing --load-balance-policy dynamic

# Launch dashboard
streamlit run src/monitoring/dashboard.py

# Run benchmarks
bash scripts/run_benchmark.sh

# Analyze results
python scripts/analyze_results.py --input-dir experiments/benchmarks
```

## Example Output

After successful training, you should see:

```
Rank 0 | Epoch 0 | Batch 0/156 | Loss: 2.3045
Rank 1 | Epoch 0 | Batch 0/78 | Loss: 2.3102
...
PERFORMANCE SUMMARY (Rank 0)
Average Iteration Time: 0.125s
Average Throughput: 256.00 samples/s
Average GPU Utilization: 85.2%
```

## Getting Help

- Check [README.md](README.md) for detailed documentation
- Open an issue on GitHub
- Review example scripts in `scripts/`

Happy training! 🚀
