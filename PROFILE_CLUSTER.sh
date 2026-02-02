#!/bin/bash
echo "=========================================="
echo "Cluster GPU Profiling"
echo "=========================================="

# Activate venv
source venv/bin/activate

# Set output directory
OUTPUT_DIR="experiments/demo_nccl_heterogeneous/configs"
mkdir -p "$OUTPUT_DIR"

echo ""
echo "Profiling this node's GPUs..."
echo "Output will be saved to: $OUTPUT_DIR"
echo ""

# Run profiler
python -m src.profiling.main --output-dir "$OUTPUT_DIR"

echo ""
echo "=========================================="
echo "✅ Profiling Complete!"
echo "=========================================="
echo ""
echo "Generated files:"
ls -lh "$OUTPUT_DIR"

echo ""
echo "Next steps:"
echo "  1. Copy gpu_profiles.json to all worker nodes"
echo "  2. Run training with --enable-load-balancing"
echo "  3. Training will automatically adjust batch sizes based on GPU profiles"
echo ""
