# GNN Model Configuration Guide

## Overview
GNN (Graph Neural Network) 모델은 계층 구조를 명시적으로 활용하여 분류 성능을 향상시킵니다.

## Available Models

### 1. BERT Baseline (default)
표준 BERT encoder + Linear classifier

```yaml
model:
  model_type: "bert"
  model_name: "bert-base-uncased"
  num_classes: 531
  dropout: 0.1
```

### 2. GCN (Graph Convolutional Network)
계층 구조를 그래프로 모델링하여 정보 전파

```yaml
model:
  model_type: "gcn"
  model_name: "bert-base-uncased"
  num_classes: 531
  dropout: 0.1
  
  # GNN-specific parameters
  gnn_hidden_dim: 512
  gnn_num_layers: 2
```

### 3. GAT (Graph Attention Network)
Attention 메커니즘으로 중요한 계층 관계 학습

```yaml
model:
  model_type: "gat"
  model_name: "bert-base-uncased"
  num_classes: 531
  dropout: 0.1
  
  # GAT-specific parameters
  gnn_hidden_dim: 512
  gnn_num_layers: 2
  gnn_num_heads: 4  # Multi-head attention
```

## Model Architecture

### BERT Baseline
```
Text → BERT Encoder → [CLS] (768) → Linear → Logits (531)
```

### GCN
```
Text → BERT Encoder → [CLS] (768) → Linear → Text Logits (531)
                                              ↓
Class Embeddings (531, 512) → GCN Layers → Class Scores (531)
                                              ↓
                                    Text Logits + Class Scores
```

### GAT
```
Text → BERT Encoder → [CLS] (768) → Linear → Text Logits (531)
                                              ↓
Class Embeddings (531, 512) → GAT Layers (Multi-head) → Class Scores (531)
                                              ↓
                                    Text Logits + Class Scores
```

## Ablation Study Configuration

### Experiment 1: Model Architecture
```yaml
# Baseline
model:
  model_type: "bert"

# GCN variant
model:
  model_type: "gcn"
  gnn_num_layers: 2

# GAT variant
model:
  model_type: "gat"
  gnn_num_layers: 2
  gnn_num_heads: 4
```

### Experiment 2: GNN Depth
```yaml
# Shallow
model:
  model_type: "gcn"
  gnn_num_layers: 1

# Medium
model:
  model_type: "gcn"
  gnn_num_layers: 2

# Deep
model:
  model_type: "gcn"
  gnn_num_layers: 3
```

### Experiment 3: Hidden Dimensions
```yaml
# Small
model:
  gnn_hidden_dim: 256

# Medium
model:
  gnn_hidden_dim: 512

# Large
model:
  gnn_hidden_dim: 1024
```

### Experiment 4: GAT Attention Heads
```yaml
model:
  model_type: "gat"
  gnn_num_heads: 2  # or 4, 8
```

### Experiment 5: Loss Functions
```yaml
training:
  loss_type: "bce"  # Baseline

training:
  loss_type: "hierarchical"  # With hierarchy constraint
  lambda_hier: 0.1

training:
  loss_type: "focal"  # Class imbalance handling
  focal_gamma: 2.0

training:
  loss_type: "asymmetric"  # Multi-label specific
```

### Experiment 6: Self-Training
```yaml
# Without self-training
self_training:
  enabled: false

# With self-training
self_training:
  enabled: true
  confidence_threshold: 0.7
  max_iterations: 3
```

## Quick Experiment Switching

### Create Multiple Configs
```bash
# config/bert_baseline.yaml
model:
  model_type: "bert"
training:
  loss_type: "bce"

# config/gcn_hierarchical.yaml
model:
  model_type: "gcn"
  gnn_num_layers: 2
training:
  loss_type: "hierarchical"

# config/gat_hierarchical.yaml
model:
  model_type: "gat"
  gnn_num_layers: 2
  gnn_num_heads: 4
training:
  loss_type: "hierarchical"
```

### Run Experiments
```bash
# Experiment 1: BERT Baseline
python3 scripts/train_with_config.py --config config/bert_baseline.yaml

# Experiment 2: GCN + Hierarchical Loss
python3 scripts/train_with_config.py --config config/gcn_hierarchical.yaml

# Experiment 3: GAT + Hierarchical Loss
python3 scripts/train_with_config.py --config config/gat_hierarchical.yaml
```

### Override Parameters
```bash
# Quick parameter changes without editing config
python3 scripts/train_with_config.py \
    --config config/config.yaml \
    --model_type gcn \
    --loss_type hierarchical

python3 scripts/train_with_config.py \
    --config config/config.yaml \
    --model_type gat \
    --num_epochs 10
```

## Performance Comparison Table

### Expected Results Structure

| Model | Loss | Hierarchy | Micro F1 | Macro F1 | Training Time |
|-------|------|-----------|----------|----------|---------------|
| BERT | BCE | No | 0.XX | 0.XX | XXm |
| BERT | Hierarchical | Yes | 0.XX | 0.XX | XXm |
| GCN | BCE | No | 0.XX | 0.XX | XXm |
| GCN | Hierarchical | Yes | 0.XX | 0.XX | XXm |
| GAT | Hierarchical | Yes | 0.XX | 0.XX | XXm |
| GCN+Self-train | Hierarchical | Yes | 0.XX | 0.XX | XXm |

## Implementation Details

### GCN Layer
- Symmetric normalization: D^(-1/2) A D^(-1/2)
- Adjacency from taxonomy hierarchy
- ReLU activation between layers

### GAT Layer
- Multi-head attention mechanism
- Learned attention weights
- Attention masking with hierarchy structure

### Hierarchy Integration
- Undirected edges (parent ↔ child)
- Self-loops included
- Normalized adjacency matrix

## Tips for Ablation Studies

1. **Fix Random Seed:**
```yaml
misc:
  seed: 42
```

2. **Use Same Data:**
- Generate silver labels once
- Reuse for all experiments

3. **Save All Checkpoints:**
```yaml
evaluation:
  save_every: 1
output:
  output_dir: "models/{model_type}_{loss_type}"
```

4. **Log Everything:**
```yaml
output:
  log_dir: "logs/{experiment_name}"
```

5. **Compare Metrics:**
```python
# Compare results across experiments
import json
results = {}
for exp in ['bert_bce', 'gcn_hier', 'gat_hier']:
    with open(f'models/{exp}/metrics.json') as f:
        results[exp] = json.load(f)
```

## Troubleshooting

**OOM (Out of Memory):**
```yaml
model:
  gnn_hidden_dim: 256  # Reduce from 512
training:
  batch_size: 8  # Reduce from 16
```

**Slow Training:**
```yaml
misc:
  mixed_precision: true  # Enable FP16
```

**Poor GAT Performance:**
```yaml
model:
  gnn_num_heads: 2  # Reduce heads
  dropout: 0.3  # Increase dropout
```

## Example Ablation Script

```bash
#!/bin/bash

# Ablation study: Model architectures
for model in bert gcn gat; do
    python3 scripts/train_with_config.py \
        --config config/config.yaml \
        --model_type $model \
        --output_dir models/ablation_model_${model}
done

# Ablation study: Loss functions
for loss in bce focal hierarchical asymmetric; do
    python3 scripts/train_with_config.py \
        --config config/config.yaml \
        --loss_type $loss \
        --output_dir models/ablation_loss_${loss}
done

# Ablation study: GNN depth
for layers in 1 2 3; do
    python3 scripts/train_with_config.py \
        --config config/config.yaml \
        --model_type gcn \
        --gnn_num_layers $layers \
        --output_dir models/ablation_depth_${layers}
done
```
