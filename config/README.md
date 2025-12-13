# Configuration Guide for Ablation Studies

## Directory Structure

The project uses a dynamic directory structure to support multiple experiments:

```
models/
├── baseline/          # BERT baseline
├── gcn/              # GCN model
├── gat/              # GAT model
├── focal_loss/       # Focal loss ablation
└── asymmetric_loss/  # Asymmetric loss ablation

results/
├── predictions/
│   ├── baseline_YYYYMMDD_HHMMSS.pkl
│   ├── baseline_YYYYMMDD_HHMMSS.csv
│   ├── gcn_YYYYMMDD_HHMMSS.pkl
│   └── ...
└── images/
    ├── baseline/
    │   ├── training_loss.png
    │   └── metrics_comparison.png
    ├── gcn/
    ├── gat/
    └── ablation/     # Cross-model comparisons
        └── model_comparison_loss.png
```

## Running Different Models

### 1. BERT Baseline

```bash
python3 src/training/train_baseline.py \
    --model_type baseline \
    --loss_type bce \
    --num_epochs 5
```

Models saved to: `models/baseline/`

### 2. GCN Model

```bash
python3 src/training/train_baseline.py \
    --model_type gcn \
    --loss_type hierarchical \
    --num_epochs 5
```

Models saved to: `models/gcn/`

### 3. GAT Model

```bash
python3 src/training/train_baseline.py \
    --model_type gat \
    --loss_type hierarchical \
    --num_epochs 5
```

Models saved to: `models/gat/`

### 4. Focal Loss Ablation

```bash
python3 src/training/train_baseline.py \
    --model_type focal_loss \
    --loss_type focal \
    --num_epochs 5
```

Models saved to: `models/focal_loss/`

## Running Inference

For each trained model:

```bash
python3 src/inference/predict.py \
    --model_path models/{model_type}/best_model.pt \
    --model_name {model_type}
```

Predictions saved to: `results/predictions/{model_type}_YYYYMMDD_HHMMSS.pkl`

## Visualization

In `notebooks/Visualization.ipynb`, change the `MODEL_NAME` variable:

```python
MODEL_NAME = 'baseline'  # or 'gcn', 'gat', etc.
```

Images saved to: `results/images/{model_name}/`

For cross-model comparisons, images are saved to: `results/images/ablation/`

## Custom Configuration Files

You can create model-specific configs:

1. Copy `config/config.yaml` to `config/{model_type}.yaml`
2. Modify model-specific parameters
3. Load in training script (future enhancement)

Example:
```yaml
# config/gcn.yaml
model:
  model_type: "gcn"
  gnn_hidden_dim: 512
  gnn_num_layers: 2

output:
  output_dir: "models/gcn"
```
