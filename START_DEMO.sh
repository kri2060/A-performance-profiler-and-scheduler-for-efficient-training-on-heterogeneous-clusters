#!/bin/bash
# Quick Demo Start Script
# This shows the configuration before starting

echo "=========================================="
echo "HETEROGENEOUS CLUSTER TRAINING DEMO"
echo "=========================================="
echo ""
echo "Configuration:"
echo "  Backend: Gloo (supports CPU + GPU heterogeneous)"
echo "  Model: Simple CNN (lightweight)"
echo "  Dataset: Synthetic Images (1000 samples, 32x32)"
echo "  Batch Size: 32"
echo "  Epochs: 5"
echo "  Learning Rate: 0.01"
echo ""
echo "Cluster Setup:"
echo "  Master (Rank 0): Linux with GPU"
echo "  Worker (Rank 1): Windows WSL2 with GPU"
echo ""
echo "Features Enabled:"
echo "  ✓ Performance Profiling"
echo "  ✓ Dynamic Load Balancing"
echo "  ✓ Heterogeneous Batch Sizing"
echo ""
echo "=========================================="
echo "Starting Master Node..."
echo "=========================================="

./master.sh
