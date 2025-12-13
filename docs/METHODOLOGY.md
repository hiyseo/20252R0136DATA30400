# Methodology: Hierarchical Multi-Label Classification for Amazon Products

## Overview

This project implements a **hierarchical multi-label text classification** system for Amazon product categorization using a **hybrid approach** combining keyword matching, semantic similarity, and self-training with soft pseudo-labels.

---

## 1. Dataset Structure

### Amazon Product Dataset
- **531 classes** organized in a hierarchical taxonomy
- **DAG structure** (Directed Acyclic Graph): 25 nodes (4.7%) have multiple parents
- **3-level hierarchy**:
  - Level 0: 6 root nodes
  - Level 1: 64 mid-level categories
  - Level 2: 461 leaf nodes (87% of all classes)
- **Training**: 29,487 samples
- **Testing**: 19,658 samples

### Key Characteristics
```
Hierarchy Properties:
├── Max depth: 3 levels
├── Average branching factor: 1.07
├── Leaf node ratio: 87%
└── Multi-parent nodes: 25 (supports multiple categorization paths)
```

---

## 2. Silver Label Generation (Baseline Approach)

Since the original dataset lacks ground-truth labels, we generate **silver labels** using a **hybrid top-down approach**.

### 2.1 Keyword Matching

**Purpose**: Capture explicit category mentions in product text.

**Method**:
1. Load class-specific keywords from `class_related_keywords.txt`
2. Preprocess keywords with regex patterns (word boundaries, case-insensitive)
3. Match keywords against product text
4. Score = (matched keywords) / (total keywords for class)

**Advantages**:
- High precision for explicit mentions
- Fast computation
- Interpretable results

**Limitations**:
- Misses semantic variations
- Requires comprehensive keyword lists

### 2.2 Semantic Similarity

**Purpose**: Capture semantic relationships beyond exact keyword matches.

**Model**: `sentence-transformers/all-mpnet-base-v2`
- 768-dimensional embeddings
- Pre-trained on 1B+ sentence pairs
- Strong semantic understanding

**Method**:
1. Create class descriptions: `"{class_name}: {top_10_keywords}"`
2. Encode all 531 class descriptions (cached)
3. Encode product text
4. Compute cosine similarity between product and all classes
5. Filter by threshold (default: 0.5)

**Advantages**:
- Captures semantic similarity
- Handles paraphrases and synonyms
- Generalizes beyond training keywords

**Limitations**:
- Computationally expensive
- May produce false positives on similar but incorrect classes

### 2.3 Hybrid Score Combination

**Weighted Sum**:
```python
combined_score = keyword_weight * keyword_score + similarity_weight * semantic_score
```

**Default weights**:
- `keyword_weight = 0.3` (precision-focused)
- `similarity_weight = 0.7` (recall-focused)

**Rationale**: Semantic similarity provides broader coverage while keywords ensure precision for explicit mentions.

### 2.4 Top-Down Hierarchical Filtering

**Motivation**: 
- Simple weighted sum ignores hierarchy structure
- DAG structure with multi-parent nodes requires path-aware filtering
- Shallow hierarchy (3 levels) minimizes error propagation

**Algorithm**:
```python
1. Compute combined scores for ALL classes
2. For each level (0 → 1 → 2):
   a. Select nodes at current level with score ≥ threshold
   b. Apply level-specific thresholds:
      - Root (level 0): threshold * 0.7 (more lenient)
      - Mid (level 1): threshold (standard)
      - Leaf (level 2): min_confidence (final filter)
   c. Add selected nodes' children to next level's candidate set
3. Return only nodes that pass filtering at each level
```

**Level-Specific Thresholds**:
- `root_threshold = 0.15 * 0.7 = 0.105` (lenient to avoid early pruning)
- `mid_threshold = 0.15`
- `leaf_threshold = 0.1` (min_confidence)

**Benefits**:
- Respects hierarchical structure
- Supports multi-parent nodes (all parent paths considered)
- Reduces false positives by requiring parent support
- Minimizes error accumulation (only 3 levels)

**Example**:
```
Product: "Bluetooth wireless headphones"

Step 1 - Combined Scores:
├── Electronics: 0.85 ✓
├── Audio: 0.78 ✓
├── Headphones: 0.92 ✓
├── Accessories: 0.72 ✓
├── Kitchen: 0.15 ✗ (not selected)
└── Furniture: 0.05 ✗

Step 2 - Top-Down Filtering:
Level 0 (Roots):
├── Electronics: 0.85 ≥ 0.105 ✓ → Allow children: [Audio, Computers, ...]
└── Accessories: 0.72 ≥ 0.105 ✓ → Allow children: [Headphones, ...]

Level 1 (Mid):
└── Audio: 0.78 ≥ 0.15 ✓ → Allow children: [Headphones, Speakers, ...]

Level 2 (Leaf):
└── Headphones: 0.92 ≥ 0.1 ✓

Final Labels: {Electronics, Audio, Headphones, Accessories}
```

### 2.5 Silver Label Statistics

**Coverage**: ~85-95% of samples receive at least one label
**Average labels per sample**: 2-3 classes
**Label source breakdown**:
- Keyword only: ~30%
- Semantic only: ~40%
- Both methods: ~30%

---

## 3. Model Architecture

### 3.1 Text Encoder (Baseline)

**Base Model**: `bert-base-uncased`
- 12 transformer layers
- 768 hidden dimensions
- 110M parameters

**Architecture**:
```python
class TextEncoder(nn.Module):
    def __init__(self, num_classes=531, dropout=0.1):
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, 531)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        pooled = outputs.pooler_output  # [CLS] token
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 531)
        return logits
```

**Input**: Tokenized product text (max 128 tokens)
**Output**: Logits for 531 classes (multi-label, not mutually exclusive)

### 3.2 Loss Functions

#### BCE (Binary Cross-Entropy) - Baseline
```python
loss = -[y * log(σ(x)) + (1-y) * log(1-σ(x))]
```
- Treats each class independently
- Standard for multi-label classification

#### Focal Loss - Handling Class Imbalance
```python
focal_weight = (1 - pt)^γ
loss = -α * focal_weight * BCE
```
- `γ=2`: Focus on hard examples
- `α=0.25`: Balance positive/negative samples
- Reduces loss contribution from easy examples

#### Asymmetric Loss - Negative Sample Focus
```python
pos_loss = y * log(p) * (1-p)^γ_pos
neg_loss = (1-y) * log(1-p) * p^γ_neg
```
- `γ_neg=4`: Strong focus on hard negatives
- `γ_pos=1`: Standard focus on positives
- Addresses severe class imbalance (531 classes)

---

## 4. Self-Training with Soft Pseudo-Labels

### 4.1 Motivation

**Problem**: Silver labels are noisy and incomplete
**Solution**: Use model's own predictions on unlabeled data to iteratively improve

**Why Soft Labels?**
- **Hard labels** (`[0, 1, 0, 1]`): Binary decisions, loss of uncertainty information
- **Soft labels** (`[0.02, 0.87, 0.15, 0.92]`): Preserve probability distribution
- **Benefits for hierarchical multi-label**:
  - Maintains relative confidence across multiple labels
  - Captures uncertainty in DAG structure
  - Enables smoother learning via self-distillation

### 4.2 Self-Training Algorithm

```python
for iteration in range(max_iterations=3):
    # 1. Generate pseudo-labels on unlabeled data
    model.eval()
    probs = sigmoid(model(unlabeled_data))  # Soft probabilities
    
    # 2. Filter confident predictions
    confident_samples = probs.max(dim=1) ≥ confidence_threshold (0.7)
    pseudo_labels = probs[confident_samples]  # Keep probabilities!
    
    # 3. Combine with labeled data
    combined_data = labeled_data + pseudo_labeled_data
    
    # 4. Train with KL Divergence loss
    model.train()
    for batch in combined_data:
        if is_pseudo_labeled:
            loss = KLD(log(σ(logits)), soft_targets)
        else:
            loss = BCE(logits, hard_labels)
    
    # 5. Check convergence
    if improvement < min_improvement:
        break
```

### 4.3 KL Divergence Loss for Soft Labels

**Standard BCE**: Requires binary targets `{0, 1}`
**KLD**: Accepts probability distributions `[0, 1]`

```python
KLD(P||Q) = Σ P(x) * log(P(x) / Q(x))

P = soft pseudo-labels (teacher)
Q = model predictions (student)
```

**Implementation**:
```python
class KLDivLossForSoftLabels(nn.Module):
    def forward(self, logits, soft_targets):
        log_probs = F.logsigmoid(logits)
        soft_targets = soft_targets.clamp(min=1e-7, max=1.0)
        return F.kl_div(log_probs, soft_targets, reduction='batchmean')
```

**Advantages**:
1. **Self-distillation**: Model learns from its own softened predictions
2. **Uncertainty preservation**: Maintains confidence scores (e.g., 0.85 vs 0.55)
3. **Multi-label awareness**: Preserves relative probabilities across 531 classes
4. **Hierarchy respect**: Soft labels naturally encode parent-child relationships

**Example**:
```
Hard label:  [0, 1, 0, 1, 0] → Loss focuses only on correct/incorrect
Soft label:  [0.02, 0.87, 0.15, 0.92, 0.08]
             ↑ Low confidence negative
                  ↑ High confidence positive
                       ↑ Borderline (maybe related?)
                            ↑ Very confident positive
                                  ↑ Low confidence negative
→ Loss considers ALL probability distributions, smoother gradients
```

### 4.4 Confidence Threshold Selection

**Threshold = 0.7**
- Too low (< 0.5): Noisy pseudo-labels degrade performance
- Too high (> 0.9): Too few pseudo-labels, limited benefit
- 0.7: Balance between quality and quantity

**Filtering Strategy**:
- Keep samples where `max(probs) ≥ 0.7`
- Ensures at least one confident prediction
- Typical retention: 40-60% of unlabeled data

---

## 5. Training Pipeline

### 5.1 Standard Training (Without Self-Training)

```bash
python src/training/train_baseline.py \
  --model_type baseline \
  --batch_size 16 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --loss_type bce \
  --train_labels_path data/intermediate/train_silver_labels.pkl
```

**Steps**:
1. Load silver labels (hybrid top-down generated)
2. Create dataloaders with BERT tokenization
3. Initialize TextEncoder (BERT + classifier head)
4. Train with BCE loss for 3 epochs
5. Save best model based on validation F1

### 5.2 Self-Training Mode

```bash
python src/training/train_baseline.py \
  --model_type baseline_self_training \
  --use_self_training \
  --self_training_confidence 0.7 \
  --self_training_iterations 3 \
  --num_epochs 2 \
  --loss_type kld
```

**Steps per iteration**:
1. **Initial training**: Train on silver-labeled data with BCE
2. **Pseudo-labeling**: Generate soft labels on test data (confidence ≥ 0.7)
3. **Data combination**: Merge labeled + pseudo-labeled samples
4. **Retraining**: Train combined data with KLD loss
5. **Convergence check**: Stop if loss improvement < 0.001

**Typical behavior**:
- Iteration 1: +2000-3000 pseudo-labeled samples
- Iteration 2: +1500-2500 samples (model more selective)
- Iteration 3: +1000-1500 samples
- Converges after 2-3 iterations

---

## 6. Evaluation Metrics

### Multi-Label Metrics

**Precision, Recall, F1** (Micro & Macro):
```python
Micro: Aggregate all classes, compute metrics globally
Macro: Compute per-class, average across classes
```

**Optimal Threshold Search**:
- Evaluate thresholds from 0.1 to 0.9
- Select threshold maximizing Micro-F1
- Typical optimal: 0.3-0.5

### Hierarchical Metrics

**Hierarchy Violation Rate**:
```python
For each predicted child class:
  if parent class not predicted:
    count as violation

violation_rate = violations / total_predictions
```

**Level-wise Accuracy**:
- Separate evaluation for root, mid, and leaf nodes
- Identifies which hierarchy levels are challenging

---

## 7. Key Design Decisions

### Why Hybrid Top-Down?

| Approach | Pros | Cons | Fit for Dataset |
|----------|------|------|----------------|
| **Simple Sum** | Easy, considers all classes | Ignores hierarchy | ❌ 87% leaf nodes misses structure |
| **Pure Top-Down** | Respects hierarchy | Error propagation | ⚠️ Risky with noisy silver labels |
| **Hybrid** | Structure + flexibility | Complex | ✅ Best for 3-level DAG |

### Why Soft Labels for Self-Training?

| Approach | Information | Suitable For | Our Choice |
|----------|-------------|--------------|------------|
| **Hard Labels** | Binary (0/1) | Single-label, confident data | ❌ |
| **Soft Labels + KLD** | Probability distribution | Multi-label, uncertain data | ✅ |

**Rationale**: 
- Multi-label classification (2-3 labels/sample)
- DAG hierarchy (relative probabilities matter)
- Noisy silver labels (uncertainty is informative)

### Why 3-Level Threshold Strategy?

```
Root (0.105) < Mid (0.15) < Leaf (0.1)
       ↑            ↑           ↑
  More lenient  Standard   Final filter
```

**Reasoning**:
- **Root**: Avoid early pruning (6 roots, broad categories)
- **Mid**: Standard filtering (64 categories, specific but not final)
- **Leaf**: Strictest (461 classes, final predictions matter most)

---

## 8. Implementation Files

```
src/
├── silver_labeling/
│   ├── generate_silver_labels.py    # Hybrid top-down labeling
│   ├── graph_utils.py               # Hierarchy operations
│   └── llm_keyword_expansion.py     # Keyword augmentation (optional)
│
├── models/
│   ├── encoder.py                   # BERT-based TextEncoder
│   ├── classifier.py                # Multi-label classifier head
│   └── gnn_classifier.py            # GNN variants (future)
│
├── training/
│   ├── train_baseline.py            # Main training script
│   ├── self_training.py             # Self-training with soft labels
│   └── loss_functions.py            # BCE, Focal, Asymmetric, KLD
│
├── inference/
│   └── predict.py                   # Inference with threshold tuning
│
└── utils/
    ├── metrics.py                   # Evaluation metrics
    ├── taxonomy_mapping.py          # Hierarchy utilities
    └── seed.py                      # Reproducibility
```

---

## 9. Reproducibility

### Random Seeds
```python
set_seed(42)  # Python, NumPy, PyTorch, CUDA
```

### Hardware Requirements
- **Minimum**: 16GB RAM, GPU with 8GB VRAM
- **Recommended**: 32GB RAM, GPU with 16GB VRAM (for batch_size=32)
- **Training time**: ~2-3 hours per iteration on V100

### Dependencies
```
python >= 3.10
torch >= 2.0.0
transformers >= 4.30.0
sentence-transformers >= 2.2.0
networkx >= 3.0
```

---

## 10. Expected Results

### Silver Label Quality
- **Coverage**: 85-95% labeled samples
- **Average labels**: 2-3 per sample
- **Hierarchy consistency**: ~90% (root-to-leaf paths valid)

### Model Performance (Baseline)
- **Micro-F1**: 0.55-0.65
- **Macro-F1**: 0.35-0.45
- **Hierarchy violation rate**: < 10%

### Self-Training Improvement
- **Micro-F1**: +3-5% absolute improvement
- **Coverage**: +10-15% more classes predicted
- **Convergence**: 2-3 iterations

---

## 11. Limitations and Future Work

### Current Limitations
1. **Silver labels are noisy**: Keyword/semantic matching has false positives
2. **No validation set**: Optimal threshold found on test set (should be separate)
3. **Class imbalance**: 87% leaf nodes dominates training
4. **Computational cost**: Semantic similarity encoding is expensive

### Future Improvements
1. **GNN-based models**: Explicitly encode hierarchy in model architecture
2. **Hierarchical loss functions**: Penalize ancestor violations
3. **Active learning**: Human annotation of high-uncertainty samples
4. **Multi-task learning**: Joint training with hierarchy prediction task
5. **Contrastive learning**: Learn better representations respecting hierarchy

---

## References

1. **TaxoClass**: Hierarchical text classification using taxonomy structure
2. **Sentence-BERT**: Semantic similarity with transformers
3. **Self-Training**: Pseudo-labeling for semi-supervised learning
4. **Knowledge Distillation**: Soft labels for model training
5. **Focal Loss**: Handling class imbalance in object detection

---

## Quick Start

### 1. Generate Silver Labels
```bash
python src/silver_labeling/generate_silver_labels.py
```

### 2. Train Baseline Model
```bash
python src/training/train_baseline.py \
  --model_type baseline \
  --num_epochs 3 \
  --batch_size 16
```

### 3. Train with Self-Training
```bash
python src/training/train_baseline.py \
  --model_type baseline_self_training \
  --use_self_training \
  --self_training_confidence 0.7 \
  --num_epochs 2
```

### 4. Run Inference
```bash
python src/inference/predict.py \
  --model_path models/baseline/best_model.pt \
  --model_name baseline
```

### 5. Analyze Results
```bash
jupyter notebook notebooks/Ablation_Analysis.ipynb
jupyter notebook notebooks/CaseStudy.ipynb
```
