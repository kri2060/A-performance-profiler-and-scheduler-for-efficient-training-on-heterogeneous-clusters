#!/usr/bin/env python3
"""Quick GPU test to verify CUDA training works"""

import torch
import torch.nn as nn
import time

print("=" * 60)
print("GPU Training Test")
print("=" * 60)

# Check CUDA
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA device count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")

# Simple model
model = nn.Sequential(
    nn.Conv2d(3, 64, 3, padding=1),
    nn.ReLU(),
    nn.Conv2d(64, 128, 3, padding=1),
    nn.ReLU(),
    nn.AdaptiveAvgPool2d(1),
    nn.Flatten(),
    nn.Linear(128, 10)
)

# Move to GPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"\nUsing device: {device}")
model = model.to(device)

# Create dummy data
batch_size = 32
x = torch.randn(batch_size, 3, 32, 32).to(device)
y = torch.randint(0, 10, (batch_size,)).to(device)

# Training step
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

print("\nRunning 10 training iterations...")
start_time = time.time()

for i in range(10):
    optimizer.zero_grad()
    output = model(x)
    loss = criterion(output, y)
    loss.backward()
    optimizer.step()

    if i % 2 == 0:
        print(f"Iteration {i}: Loss = {loss.item():.4f}")

elapsed = time.time() - start_time
print(f"\nTotal time: {elapsed:.2f}s")
print(f"Average time per iteration: {elapsed/10:.3f}s")

# Check GPU memory
if torch.cuda.is_available():
    print(f"\nGPU Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.1f} MB")
    print(f"GPU Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.1f} MB")

print("\n" + "=" * 60)
print("Test complete! Check nvidia-smi for GPU usage.")
print("=" * 60)
