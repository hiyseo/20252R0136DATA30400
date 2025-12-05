#!/bin/bash
# AWS SageMaker Training Script
# This script runs the full training pipeline on SageMaker

set -e  # Exit on error

echo "=== Starting Training Pipeline ==="
echo "Date: $(date)"
echo "Host: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo ""

# Activate conda environment if needed
# source activate pytorch_p310

# Set Python path
export PYTHONPATH="${PYTHONPATH}:/opt/ml/code"

# Training parameters (can be overridden by SageMaker)
MODEL_NAME=${MODEL_NAME:-"bert-base-uncased"}
BATCH_SIZE=${BATCH_SIZE:-16}
NUM_EPOCHS=${NUM_EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
MAX_LENGTH=${MAX_LENGTH:-128}

echo "=== Configuration ==="
echo "Model: $MODEL_NAME"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $NUM_EPOCHS"
echo "Learning rate: $LEARNING_RATE"
echo "Max length: $MAX_LENGTH"
echo ""

# Check if silver labels exist, if not generate them
if [ ! -f "data/intermediate/train_silver_labels.pkl" ]; then
    echo "=== Generating Silver Labels ==="
    python3 src/silver_labeling/generate_silver_labels.py
    echo "✓ Silver labels generated"
    echo ""
fi

# Train baseline model
echo "=== Training Baseline Model ==="
python3 src/training/train_baseline.py \
    --model_name "$MODEL_NAME" \
    --batch_size "$BATCH_SIZE" \
    --num_epochs "$NUM_EPOCHS" \
    --learning_rate "$LEARNING_RATE" \
    --max_length "$MAX_LENGTH" \
    --output_dir "models/baseline" \
    --save_every 1

echo ""
echo "=== Training Complete ==="
echo "Date: $(date)"
