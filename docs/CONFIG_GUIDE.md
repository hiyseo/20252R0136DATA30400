# Config File Usage Guide

## config.yaml 주요 설정 설명

### ✅ 수정 완료된 설정

| 설정 | 이전 값 | 수정된 값 | 이유 |
|-----|--------|----------|------|
| `model.model_type` | `bert` | `baseline` | 디렉토리 이름과 일치 |
| `training.loss_type` | `hierarchical` | `bce` | Baseline에 맞게 설정 |
| `training.num_epochs` | `5` | `3` | 일반적인 학습 횟수 |
| `output.output_dir` | 정적 경로 | 동적 생성 | `model_type`에서 자동 설정 |

---

## 사용 시나리오

### 1. Baseline Training (기본)

**config.yaml 설정:**
```yaml
model:
  model_type: "baseline"

training:
  loss_type: "bce"
  num_epochs: 3

self_training:
  enabled: false
```

**실행:**
```bash
python3 scripts/train_with_config.py
```

**결과:**
```
models/baseline/
├── best_model.pt
├── final_model.pt
└── training_history.json
```

---

### 2. Self-Training

**config.yaml 수정:**
```yaml
model:
  model_type: "baseline_self_training"

self_training:
  enabled: true
  confidence_threshold: 0.7
  max_iterations: 3
```

**실행:**
```bash
python3 scripts/train_with_config.py
```

**결과:**
```
models/baseline_self_training/
├── best_model.pt
└── training_history.json
```

---

### 3. Focal Loss

**config.yaml 수정:**
```yaml
model:
  model_type: "focal_loss"

training:
  loss_type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
```

**실행:**
```bash
python3 scripts/train_with_config.py
```

---

### 4. Command Line Override

**config는 그대로 두고 명령어로 override:**

```bash
# Batch size 변경
python3 scripts/train_with_config.py --batch_size 32

# Model 변경
python3 scripts/train_with_config.py --model_name roberta-base

# Self-training 활성화
python3 scripts/train_with_config.py --use_self_training

# 여러 개 동시 변경
python3 scripts/train_with_config.py \
  --batch_size 32 \
  --num_epochs 5 \
  --use_self_training
```

---

## 주요 파라미터 설명

### Silver Labeling
```yaml
silver_labeling:
  use_topdown_filtering: true    # Hybrid top-down approach
  keyword_weight: 0.3            # Keyword 가중치
  similarity_weight: 0.7         # Semantic similarity 가중치
  min_confidence: 0.1            # 최소 confidence (leaf level)
  topdown_threshold: 0.15        # Top-down 필터링 threshold
```

### Model
```yaml
model:
  model_type: "baseline"         # 디렉토리 이름 (baseline, gcn, gat, focal_loss 등)
  model_name: "bert-base-uncased"  # Hugging Face model
  dropout: 0.1                   # Dropout 비율
```

### Training
```yaml
training:
  batch_size: 16                 # GPU 메모리에 따라 조정 (8, 16, 32)
  num_epochs: 3                  # 일반적으로 3-5
  learning_rate: 2.0e-5          # BERT 학습률
  loss_type: "bce"               # bce, focal, asymmetric, kld
```

### Self-Training
```yaml
self_training:
  enabled: false                 # true로 변경 시 self-training 활성화
  confidence_threshold: 0.7      # Pseudo-label confidence
  max_iterations: 3              # 최대 반복 횟수
```

---

## 실험별 Config 템플릿

### Experiment 1: Baseline
```yaml
model:
  model_type: "baseline"
training:
  loss_type: "bce"
  num_epochs: 3
self_training:
  enabled: false
```

### Experiment 2: Focal Loss
```yaml
model:
  model_type: "focal_loss"
training:
  loss_type: "focal"
  num_epochs: 3
self_training:
  enabled: false
```

### Experiment 3: Self-Training
```yaml
model:
  model_type: "baseline_self_training"
training:
  loss_type: "bce"
  num_epochs: 2
self_training:
  enabled: true
  confidence_threshold: 0.7
  max_iterations: 3
```

### Experiment 4: RoBERTa
```yaml
model:
  model_name: "roberta-base"
  model_type: "roberta_baseline"
training:
  loss_type: "bce"
  num_epochs: 3
```

---

## 동적 경로 생성

`train_with_config.py`는 자동으로 `model_type`을 사용하여 경로를 생성합니다:

```python
# config.yaml
model:
  model_type: "baseline"

# 자동 생성되는 경로
output_dir: "models/baseline"
images_dir: "results/images/baseline"
```

만약 `model_type`을 변경하면:
```yaml
model:
  model_type: "focal_loss"

# 자동 생성
output_dir: "models/focal_loss"
images_dir: "results/images/focal_loss"
```

---

## 주의사항

### ⚠️ 주의할 설정

1. **`loss_type: "hierarchical"`**
   - 현재 baseline에서는 구현되지 않음
   - `bce`, `focal`, `asymmetric`만 사용

2. **`num_workers`**
   - 로컬: 0-2 (MPS는 multiprocessing 이슈)
   - AWS: 4-8

3. **`batch_size`**
   - GPU 메모리 부족 시 줄이기 (16 → 8)
   - 여유 있으면 늘리기 (16 → 32)

4. **`self_training.enabled`**
   - `true`로 설정 시 학습 시간 2배 증가
   - Baseline 먼저 확인 후 사용 권장

---

## 빠른 테스트

### 로컬 테스트 (1 epoch)
```bash
python3 scripts/train_with_config.py --num_epochs 1 --batch_size 8
```

### AWS 전체 학습
```bash
python3 scripts/train_with_config.py
# config.yaml의 모든 설정 사용
```

### 특정 실험
```bash
# Focal loss 실험
python3 scripts/train_with_config.py \
  --config config/config.yaml

# config.yaml에서 model_type: focal_loss, loss_type: focal 설정
```
