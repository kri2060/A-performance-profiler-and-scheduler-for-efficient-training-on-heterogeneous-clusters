"""
Demo Training Script: ResNet50 + CIFAR-10 + Simulated Heterogeneous DBS
========================================================================
Runs entirely on a single GPU (or CPU) — no multi-node setup required.
Simulates 4 heterogeneous workers with different compute speeds, injects
a synthetic straggler at epoch 3 so the Adaptive Load Balancer is forced
to redistribute batch partitions. Produces rank_*_metrics.json logs ready
for the Streamlit dashboard Demo Playback.

Usage:
    python run_demo_resnet_cifar10.py

Outputs:
    experiments/demo_resnet_cifar10_dbs/logs/rank_0_metrics.json
    experiments/demo_resnet_cifar10_dbs/logs/rank_1_metrics.json
    experiments/demo_resnet_cifar10_dbs/logs/rank_2_metrics.json
    experiments/demo_resnet_cifar10_dbs/logs/rank_3_metrics.json
    experiments/demo_resnet_cifar10_dbs/configs/gpu_profiles.json
"""

import sys
import os
import json
import time
import random
import logging
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# ── Resolve src path ─────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))

from scheduling.load_balancer import AdaptiveLoadBalancer
from scheduling.health_monitor import simulate_straggler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Configurable knobs ────────────────────────────────────────────────────────
EXPERIMENT_NAME = "demo_resnet_cifar10_dbs"
TOTAL_EPOCHS    = 10
STEPS_PER_EPOCH = 50          # 50 steps × 10 epochs = 500 data points per rank
GLOBAL_BATCH    = 128          # Split across 4 simulated workers
NUM_WORKERS     = 4
REBALANCE_EVERY = 10           # Re-run DBS partition every N steps
STRAGGLER_EPOCH = 3            # Worker-3 becomes slow starting at this epoch
STRAGGLER_SLOWDOWN = 3.0       # × slower than normal
FAULT_CHANCE    = 0.0          # simulate_straggler controlled separately below
DATA_DIR        = "./data"

# ── Simulated GPU profiles (heterogeneous cluster) ───────────────────────────
GPU_PROFILES = [
    {"device_id": 0, "compute_score": 3.0,  "total_memory_mb": 24000,
     "memory_bandwidth_gbps": 1008, "network_mbps": 2500, "hostname": "node-0-rtx4090"},
    {"device_id": 1, "compute_score": 1.5,  "total_memory_mb": 10000,
     "memory_bandwidth_gbps": 760,  "network_mbps": 1000, "hostname": "node-1-rtx3080"},
    {"device_id": 2, "compute_score": 0.6,  "total_memory_mb": 12000,
     "memory_bandwidth_gbps": 360,  "network_mbps": 1000, "hostname": "node-2-rtx3060"},
    {"device_id": 3, "compute_score": 0.4,  "total_memory_mb": 11000,
     "memory_bandwidth_gbps": 484,  "network_mbps": 1000, "hostname": "node-3-gtx1080ti"},
]


# ─────────────────────────────────────────────────────────────────────────────
# Lightweight ResNet50 model wrapper (uses torchvision)
# ─────────────────────────────────────────────────────────────────────────────
def build_resnet50(num_classes: int = 10) -> nn.Module:
    from torchvision import models
    try:
        model = models.resnet50(weights=None)
    except TypeError:
        model = models.resnet50(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ─────────────────────────────────────────────────────────────────────────────
# CIFAR-10 dataset loader
# ─────────────────────────────────────────────────────────────────────────────
def get_cifar10(data_dir: str, image_size: int = 32):
    from torchvision import datasets, transforms
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    return datasets.CIFAR10(root=data_dir, train=True,
                            download=True, transform=transform)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: simulate one training step for a given rank
# ─────────────────────────────────────────────────────────────────────────────
def simulate_step(model, data, target, optimizer, criterion, device):
    t0 = time.perf_counter()
    data, target = data.to(device), target.to(device)

    t_dl = time.perf_counter() - t0

    t1 = time.perf_counter()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, target)
    t_fwd = time.perf_counter() - t1

    t2 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t2

    t3 = time.perf_counter()
    optimizer.step()
    t_opt = time.perf_counter() - t3

    total = t_dl + t_fwd + t_bwd + t_opt
    throughput = data.size(0) / max(total, 1e-9)

    return loss.item(), total, t_dl, t_fwd, t_bwd, t_opt, throughput


# ─────────────────────────────────────────────────────────────────────────────
# Main demo runner
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=TOTAL_EPOCHS)
    parser.add_argument("--steps",  type=int, default=STEPS_PER_EPOCH)
    parser.add_argument("--batch",  type=int, default=GLOBAL_BATCH)
    parser.add_argument("--data-dir", default=DATA_DIR)
    args = parser.parse_args()

    # ── Setup output directories ──────────────────────────────────────────────
    out_dir = ROOT / "experiments" / EXPERIMENT_NAME
    logs_dir = out_dir / "logs"
    cfgs_dir = out_dir / "configs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    cfgs_dir.mkdir(parents=True, exist_ok=True)

    # Save GPU profiles for the dashboard to load
    with open(cfgs_dir / "gpu_profiles.json", "w") as f:
        json.dump(GPU_PROFILES, f, indent=2)
    log.info(f"Saved GPU profiles → {cfgs_dir}/gpu_profiles.json")

    # ── Adaptive Load Balancer ────────────────────────────────────────────────
    lb = AdaptiveLoadBalancer(policy="dynamic", rebalance_interval=REBALANCE_EVERY)
    lb.register_nodes(GPU_PROFILES)

    # Compute initial partition ratios
    partition = lb.compute_partitions(
        nodes_time=[1.0 / p["compute_score"] for p in GPU_PROFILES],
        partition_size=[0.25, 0.25, 0.25, 0.25],
        batch_size=args.batch,
    )
    log.info(f"Initial partitions: {[f'{p:.3f}' for p in partition]}")

    # ── Device & Model ────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Training device: {device}")

    model     = build_resnet50(num_classes=10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
                                momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss().to(device)

    # ── Dataset ───────────────────────────────────────────────────────────────
    log.info(f"Loading CIFAR-10 from {args.data_dir}…")
    dataset   = get_cifar10(args.data_dir)
    loader    = torch.utils.data.DataLoader(
        dataset,
        batch_size=max(args.batch // NUM_WORKERS, 1),
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    data_iter = iter(loader)

    def next_batch():
        nonlocal data_iter
        try:
            return next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            return next(data_iter)

    # ── Straggler simulation state per rank ───────────────────────────────────
    straggler_states = [{
        "fault_wait": False, "fault_round": 0,
        "fault_wait_time": 0.0, "saved_epoch": -1,
    } for _ in range(NUM_WORKERS)]

    # ── Per-rank metric histories ─────────────────────────────────────────────
    histories = [[] for _ in range(NUM_WORKERS)]

    # ── Training simulation loop ──────────────────────────────────────────────
    global_iter = 0
    node_times  = [1.0 / p["compute_score"] for p in GPU_PROFILES]

    for epoch in range(args.epochs):
        log.info(f"━━━ Epoch {epoch + 1}/{args.epochs} ━━━")

        # Inject straggler on Worker-3 starting at STRAGGLER_EPOCH
        if epoch == STRAGGLER_EPOCH:
            log.warning(f"⚠  Injecting straggler on Worker-3 (epoch {epoch})")
            # Artificially slow down node-3
            node_times[3] = (1.0 / GPU_PROFILES[3]["compute_score"]) * STRAGGLER_SLOWDOWN

        if epoch == STRAGGLER_EPOCH + 2:
            log.info("✓  Straggler resolved — DBS redistributing load")
            node_times[3] = 1.0 / GPU_PROFILES[3]["compute_score"]

        for step in range(args.steps):
            # Re-compute partition if rebalance interval hit
            if step % REBALANCE_EVERY == 0:
                partition = lb.compute_partitions(
                    nodes_time=node_times,
                    partition_size=list(partition),
                    batch_size=args.batch,
                )
                log.debug(f"  DBS partition: {[f'{p:.3f}' for p in partition]}")

            # One real forward/backward pass (shared model for simplicity)
            data, target = next_batch()
            loss_val, t_total, t_dl, t_fwd, t_bwd, t_opt, base_tput = \
                simulate_step(model, data, target, optimizer, criterion, device)

            # Simulate each worker's perspective
            for rank in range(NUM_WORKERS):
                speed  = GPU_PROFILES[rank]["compute_score"]
                noise  = random.gauss(1.0, 0.03)
                # Scale timings: slower workers take proportionally longer
                scale  = node_times[rank] / node_times[0]

                # Inject straggler delay for worker 3 during straggler window
                extra = 0.0
                if epoch >= STRAGGLER_EPOCH and epoch < STRAGGLER_EPOCH + 2 and rank == 3:
                    extra = random.uniform(0.04, 0.08)

                w_total = t_total * scale * noise + extra
                w_sync  = max(0.0, w_total - (t_fwd + t_bwd + t_opt) * scale * noise)
                w_tput  = (partition[rank] * args.batch) / max(w_total, 1e-9)
                batch_for_rank = max(1, int(partition[rank] * args.batch))

                # Simulate GPU util: busier for faster GPUs
                base_util = min(95, int(speed * 30) + random.randint(-5, 5))
                gpu_mem   = 40 + random.uniform(-3, 5)

                record = {
                    "epoch":             epoch,
                    "iteration":         global_iter * NUM_WORKERS + rank,
                    "loss":              loss_val + random.gauss(0, 0.05),
                    "batch_size":        batch_for_rank,
                    "throughput":        w_tput,
                    "iteration_time":    w_total,
                    "data_loading_time": t_dl * scale * noise,
                    "forward_time":      t_fwd * scale * noise,
                    "backward_time":     t_bwd * scale * noise,
                    "optimizer_time":    t_opt * scale * noise,
                    "sync_time":         w_sync,
                    "gpu_utilization":   base_util,
                    "gpu_memory_percent": gpu_mem,
                    "is_straggler":      (
                        1 if (rank == 3 and epoch >= STRAGGLER_EPOCH
                              and epoch < STRAGGLER_EPOCH + 2)
                        else 0
                    ),
                    "partition_ratio":   float(partition[rank]),
                }
                histories[rank].append(record)

                # Update load balancer with this rank's timing
                lb.update_node_stats(rank, {
                    "utilization":    base_util,
                    "memory_percent": gpu_mem,
                    "iteration_time": w_total,
                })

            global_iter += 1

            if step % 10 == 0:
                bs_str = " | ".join([f"W{r}={int(partition[r]*args.batch)}"
                                     for r in range(NUM_WORKERS)])
                log.info(f"  Epoch {epoch} step {step:3d} | loss={loss_val:.4f} | batches: {bs_str}")

        # Flush all rank logs every epoch for dashboard live-view
        for rank in range(NUM_WORKERS):
            p = logs_dir / f"rank_{rank}_metrics.json"
            with open(p, "w") as f:
                json.dump(histories[rank], f, indent=2)

        log.info(f"  Saved {len(histories[0])} records/rank → {logs_dir}")

    # ── Final save ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info(f"Training complete! Logs at: {logs_dir}")
    for rank in range(NUM_WORKERS):
        log.info(f"  rank_{rank}_metrics.json  ({len(histories[rank])} rows)")
    log.info("=" * 60)
    log.info(f"Launch dashboard: streamlit run src/monitoring/dashboard.py")


if __name__ == "__main__":
    main()
