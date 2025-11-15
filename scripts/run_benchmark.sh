#!/bin/bash

# Benchmark script for heterogeneous cluster training
# Tests different configurations and compares performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "================================"
echo "Heterogeneous Cluster Benchmark"
echo "================================"

# Configuration
MODELS=("simple_cnn" "resnet50")
DATASETS=("synthetic_image")
BATCH_SIZE=32
EPOCHS=5
NUM_SAMPLES=5000
IMAGE_SIZE=32

OUTPUT_DIR="$PROJECT_ROOT/experiments/benchmarks"
mkdir -p "$OUTPUT_DIR"

# 1. Profile GPUs first
echo ""
echo "Step 1: Profiling GPUs..."
echo "-------------------------"
python -m src.profiling.main \
    --output-dir "$OUTPUT_DIR/configs" \
    || echo "GPU profiling failed (continuing anyway)"

# 2. Baseline: No load balancing
echo ""
echo "Step 2: Baseline (No Load Balancing)..."
echo "---------------------------------------"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        EXPERIMENT_NAME="baseline_${MODEL}_${DATASET}"
        echo "Running: $EXPERIMENT_NAME"

        python -m src.training.main \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --num-samples $NUM_SAMPLES \
            --image-size $IMAGE_SIZE \
            --enable-profiling \
            --output-dir "$OUTPUT_DIR" \
            --experiment-name "$EXPERIMENT_NAME" \
            --backend gloo \
            || echo "Baseline experiment $EXPERIMENT_NAME failed"
    done
done

# 3. Proportional load balancing
echo ""
echo "Step 3: Proportional Load Balancing..."
echo "--------------------------------------"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        EXPERIMENT_NAME="proportional_${MODEL}_${DATASET}"
        echo "Running: $EXPERIMENT_NAME"

        python -m src.training.main \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --num-samples $NUM_SAMPLES \
            --image-size $IMAGE_SIZE \
            --enable-profiling \
            --enable-load-balancing \
            --load-balance-policy proportional \
            --output-dir "$OUTPUT_DIR" \
            --experiment-name "$EXPERIMENT_NAME" \
            --gpu-profiles "$OUTPUT_DIR/configs/gpu_profiles.json" \
            --backend gloo \
            || echo "Proportional experiment $EXPERIMENT_NAME failed"
    done
done

# 4. Dynamic load balancing
echo ""
echo "Step 4: Dynamic Load Balancing..."
echo "---------------------------------"
for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        EXPERIMENT_NAME="dynamic_${MODEL}_${DATASET}"
        echo "Running: $EXPERIMENT_NAME"

        python -m src.training.main \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --batch-size $BATCH_SIZE \
            --epochs $EPOCHS \
            --num-samples $NUM_SAMPLES \
            --image-size $IMAGE_SIZE \
            --enable-profiling \
            --enable-load-balancing \
            --load-balance-policy dynamic \
            --output-dir "$OUTPUT_DIR" \
            --experiment-name "$EXPERIMENT_NAME" \
            --gpu-profiles "$OUTPUT_DIR/configs/gpu_profiles.json" \
            --backend gloo \
            || echo "Dynamic experiment $EXPERIMENT_NAME failed"
    done
done

echo ""
echo "================================"
echo "Benchmark Complete!"
echo "================================"
echo "Results saved to: $OUTPUT_DIR"
echo ""
echo "To analyze results, run:"
echo "  python scripts/analyze_results.py --input-dir $OUTPUT_DIR"
echo ""
echo "To view dashboard:"
echo "  streamlit run src/monitoring/dashboard.py"
