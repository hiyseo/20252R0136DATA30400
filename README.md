# Hierarchical Multi-Label Product Classification

**Student ID:** 20252R0136  
**Course:** DATA304 - Machine Learning Applications  
**Project:** Amazon Products Hierarchical Classification with Self-Training

---

## 📋 Overview

This project implements a **2-stage hierarchical multi-label classification system** for Amazon product taxonomy:
1. **Stage 1**: Supervised learning with silver labels using BCE loss
2. **Stage 2**: Self-training with soft pseudo-labels using KL Divergence loss

**Key Features:**
- Hybrid top-down silver label generation (keyword + semantic + hierarchy filtering)
- BERT-based encoder with 531-class multi-label classifier
- Soft pseudo-label self-training for semi-supervised learning
- DAG-structured taxonomy with 3 levels (root, mid, leaf)

---

## 📂 Project Structure

```
data304_final/
├── config/
│   └── config.yaml                          # Centralized configuration
├── data/
│   ├── raw/
│   │   └── Amazon_products/                 # Original dataset
│   │       ├── train/train_corpus.txt       # 29,487 samples
│   │       ├── test/test_corpus.txt         # 19,658 samples
│   │       ├── classes.txt                  # 531 classes
│   │       ├── class_hierarchy.txt          # DAG structure
│   │       └── class_related_keywords.txt   # Keywords per class
│   ├── intermediate/                        # Generated files
│   │   ├── train_silver_labels.pkl          # Silver labels (70% coverage)
│   │   └── test_silver_labels.pkl           # For pseudo-labeling
│   └── output/                              # Processed outputs
├── src/
│   ├── data_preprocessing.py                # Data loader
│   ├── models/
│   │   ├── encoder.py                       # BERT encoder
│   │   ├── classifier.py                    # Multi-label classifier
│   │   └── gnn_classifier.py                # GNN models (GCN, GAT)
│   ├── training/
│   │   ├── train_baseline.py                # Main training script
│   │   ├── self_training.py                 # Self-training with soft labels
│   │   └── loss_functions.py                # BCE, Focal, Asymmetric, KLD
│   ├── inference/
│   │   ├── predict.py                       # Generate predictions
│   │   └── dummy_baseline.py                # Simple baseline
│   ├── silver_labeling/
│   │   ├── generate_silver_labels.py        # Hybrid top-down approach
│   │   ├── graph_utils.py                   # Hierarchy analysis
│   │   └── llm_keyword_expansion.py         # LLM-based expansion
│   └── utils/
│       ├── logger.py                        # Logging utilities
│       ├── metrics.py                       # Evaluation metrics
│       ├── seed.py                          # Random seed control
│       └── taxonomy_mapping.py              # Hierarchy utilities
├── scripts/
│   ├── generate_labels.py                   # Generate silver labels
│   ├── train_with_config.py                 # Config-based training
│   └── generate_submission.py               # Create Kaggle submission
├── notebooks/
│   ├── EDA.ipynb                            # Exploratory data analysis
│   ├── Ablation_Analysis.ipynb              # Experiment results
│   └── CaseStudy.ipynb                      # Error analysis
├── models/                                  # Trained models
│   └── baseline/
│       ├── best_model.pt                    # Final model weights
│       └── training_history.json            # Loss curves
├── results/                                 # Predictions and visualizations
│   ├── predictions/
│   │   └── baseline_YYYYMMDD_HHMMSS.csv    # Predictions
│   ├── submissions/
│   │   └── 20252R0136_baseline.csv          # Kaggle submission
│   └── images/                              # Visualizations
├── logs/                                    # Training logs
├── docs/                                    # Documentation
│   ├── CONFIG.md                            # Configuration guide
│   ├── PIPELINE.md                          # Complete pipeline guide
│   └── METHODOLOGY.md                       # Detailed methodology
├── requirements.txt                         # Python dependencies
├── run.sh                                   # Quick start script
└── README.md                                # This file
```

---

## 🚀 Quick Start

### 1. Environment Setup

```bash
# Create virtual environment
python3 -m venv data304
source data304/bin/activate  # On macOS/Linux
# data304\Scripts\activate   # On Windows

# Install dependencies
pip install -r requirements.txt
```

**Dependencies:**
- Python 3.10+
- PyTorch 2.0+
- Transformers 4.30+
- sentence-transformers
- NetworkX 3.0+
- scikit-learn, pandas, numpy

### 2. Data Preprocessing (Already Done)

```bash
python3 src/data_preprocessing.py
```

**Output:**
```
✓ Loaded 29,487 training samples
✓ Loaded 19,658 test samples
✓ 531 classes in 3-level DAG hierarchy
✓ 6 root nodes, 25 multi-parent nodes
```

### 3. Generate Silver Labels

```bash
python3 scripts/generate_labels.py
```

**Method:** Hybrid top-down approach
- **Score**: `0.3 × keyword_matching + 0.7 × semantic_similarity`
- **Filtering**: Level-by-level hierarchy filtering (root → mid → leaf)
- **Thresholds**: τ₀=0.105, τ₁=0.15, τ₂=0.1

**Output:**
```
data/intermediate/
├── train_silver_labels.pkl    # 70% coverage, 3.25 labels/sample
└── test_silver_labels.pkl     # For pseudo-labeling
```

### 4. Train Baseline Model

```bash
# Config-based training (recommended)
python3 scripts/train_with_config.py

# Or direct command
python3 src/training/train_baseline.py \
  --model_type baseline \
  --num_epochs 2 \
  --use_self_training
```

**Training Process:**
```
Stage 1: BCE Initialization (2 epochs)
  └─ Train on silver labels with BCE loss
  └─ Purpose: Initialize model for pseudo-labeling

Stage 2: Self-Training (3 iterations)
  └─ Generate soft pseudo-labels (confidence ≥ 0.7)
  └─ Train with KLD loss on labeled + pseudo-labeled data
  └─ Purpose: Refine model with unlabeled data
```

**Expected time:** 3-4 hours on V100 GPU

**Output:**
```
models/baseline/
├── best_model.pt              # Final trained model
├── training_history.json      # Loss curves
└── checkpoint_epoch_*.pt      # Intermediate checkpoints
```

### 5. Generate Predictions

```bash
python3 src/inference/predict.py \
  --model_path models/baseline/best_model.pt \
  --model_name baseline
```

**Output:**
```
results/predictions/
├── baseline_20251213_143022.pkl    # For analysis
└── baseline_20251213_143022.csv    # For submission
```

### 6. Create Submission

```bash
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_*.csv \
  --output results/submissions/20252R0136_baseline.csv
```

**Format:**
```csv
20252R0136,pid
20252R0136,50 125 328
20252R0136,12 89 245 401
...
```

---

## 📊 Baseline Methodology

### Stage 0: Silver Label Generation

**Mathematical Formulation:**

$$s(x_i, c_j) = 0.3 \cdot \frac{|\text{tokens}(x_i) \cap K_{c_j}|}{|K_{c_j}|} + 0.7 \cdot \cos(\phi(x_i), \phi(c_j))$$

where:
- $\phi(\cdot)$ = sentence-transformers embedding (all-mpnet-base-v2)
- $K_{c_j}$ = keywords for class $c_j$

**Top-Down Filtering:**
```
For each level ℓ ∈ {0, 1, 2}:
  1. Select: Selected_ℓ = {c : s(x, c) ≥ τ_ℓ}
  2. Allow children: Allowed_{ℓ+1} = {c : parent(c) ∈ Selected_ℓ}
```

### Stage 1: Supervised Learning (BCE)

**Loss Function:**

$$L_{\text{BCE}}(x, y) = -\frac{1}{k}\sum_{j=1}^{k} [y_j \log p_j + (1-y_j) \log(1-p_j)]$$

where $y \in \{0, 1\}^k$ are binary silver labels.

### Stage 2: Self-Training (KLD)

**Pseudo-Label Generation:**

$$\tilde{p} = \sigma(f_\theta(x)) \quad \text{if} \quad \max(\tilde{p}) \geq 0.7$$

**Loss Function:**

$$L_{\text{KLD}}(x, \tilde{p}) = \frac{1}{k}\sum_{j=1}^{k} \tilde{p}_j \log\frac{\tilde{p}_j}{p_j}$$

where $\tilde{p} \in [0, 1]^k$ are soft pseudo-labels.

**Key Difference:**
- BCE uses **hard labels** (0/1) → Forces binary decisions
- KLD uses **soft labels** (0~1) → Preserves uncertainty

---

## ⚙️ Configuration

Edit `config/config.yaml` to change settings:

```yaml
model:
  model_type: "baseline"          # Experiment identifier
  model_name: "bert-base-uncased"

training:
  batch_size: 16
  num_epochs: 2                   # Stage 1 initialization
  learning_rate: 2.0e-5
  loss_type: "bce"                # Stage 1: BCE, Stage 2: KLD (auto)

self_training:
  enabled: true                   # Enable 2-stage training
  confidence_threshold: 0.7
  max_iterations: 3

output:
  output_dir: "models/{model_type}"  # Auto-resolved placeholder
```

See `docs/CONFIG.md` for detailed configuration guide.

---

## 📈 Performance

### Silver Label Statistics
- **Coverage**: 70.0% (20,640/29,487 training samples)
- **Avg labels/sample**: 3.25
- **Class usage**: 445/531 (83.8%)

### Model Architecture
- **Encoder**: BERT-base-uncased (109.9M parameters)
- **Classifier**: Linear(768 → 531)
- **Total parameters**: ~110M

### Expected Training Performance
- **Stage 1 BCE loss**: 0.60-0.65
- **Stage 2 KLD loss**: 0.35-0.40
- **Pseudo-label coverage**: 75-85% of test set

---

## 🛠️ Advanced Usage

### Ablation Studies

**Experiment 1: No Self-Training**
```yaml
# config.yaml
model:
  model_type: "no_self_training"
self_training:
  enabled: false
training:
  num_epochs: 5
```

**Experiment 2: Focal Loss**
```yaml
model:
  model_type: "focal_loss"
training:
  loss_type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
self_training:
  enabled: false
```

### AWS SageMaker Deployment

```bash
# SSH to SageMaker instance
ssh -i your-key.pem ubuntu@your-instance

# Clone and setup
git clone https://github.com/hiyseo/20252R0136DATA30400.git
cd 20252R0136DATA30400
source data304/bin/activate

# Run pipeline
python3 scripts/generate_labels.py
python3 scripts/train_with_config.py
python3 src/inference/predict.py \
  --model_path models/baseline/best_model.pt \
  --model_name baseline
```

---

## 📚 Documentation

- **`docs/CONFIG.md`**: Configuration parameters and examples
- **`docs/PIPELINE.md`**: Complete step-by-step pipeline guide
- **`docs/METHODOLOGY.md`**: Detailed methodology and mathematical formulation

---

## 🔍 Analysis

Run Jupyter notebooks for detailed analysis:

```bash
jupyter notebook notebooks/EDA.ipynb              # Dataset exploration
jupyter notebook notebooks/Ablation_Analysis.ipynb  # Experiment comparison
jupyter notebook notebooks/CaseStudy.ipynb         # Error analysis
```

---

## 🐛 Troubleshooting

### CUDA Out of Memory
```yaml
# config.yaml
training:
  batch_size: 8  # Reduce from 16
data:
  max_length: 64  # Reduce from 128
```

### Training Too Slow
```yaml
training:
  num_epochs: 2  # Reduce epochs
self_training:
  max_iterations: 2  # Reduce iterations
```

### Low Coverage
```yaml
silver_labeling:
  topdown_threshold: 0.1  # Lower threshold (was 0.15)
  min_confidence: 0.05    # Lower confidence (was 0.1)
```

---

## ⚖️ License

This project is for academic purposes only (DATA304 Course Project).
