#!/bin/bash
# Quick test to verify everything works before uploading to SageMaker

echo "=== Quick Pipeline Test ==="

# Run silver label generation test
echo "1. Testing silver label generation..."
python3 src/silver_labeling/generate_silver_labels.py

# Run training pipeline test
echo ""
echo "2. Testing training pipeline..."
python3 scripts/test_training_pipeline.py

echo ""
echo "=== All Tests Complete ==="
