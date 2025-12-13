# Complete Pipeline Guide - Baseline Model

## 파이프라인 단계별 명령어

### 0. 환경 준비 (최초 1회)

```bash
cd /Users/yoonseo/Desktop/data304_final

# 가상환경 활성화
source data304/bin/activate

# 의존성 설치 확인
pip install -r requirements.txt
```

---

## 🔹 STEP 1: Silver Label 생성 (Hybrid Top-Down)

### 명령어
```bash
python3 src/silver_labeling/generate_silver_labels.py
```

**또는 config 기반:**
```bash
python3 scripts/generate_labels.py
```

### 설정
- `config/config.yaml`의 `silver_labeling` 섹션 참고
- **Hybrid approach**: keyword (30%) + semantic (70%) + top-down filtering
- **Top-down threshold**: 0.15 (Root: 0.105, Mid: 0.15, Leaf: 0.1)

### 생성 파일
```
data/intermediate/
├── train_silver_labels.pkl
└── test_silver_labels.pkl
```

### 예상 결과
- Coverage: ~70-85%
- Average labels per sample: 2-4
- 소요 시간: 5-10분 (로컬), 3-5분 (AWS)

---

## 🔹 STEP 2: Baseline 모델 학습

### Option A: 직접 명령어 (추천)

```bash
python3 src/training/train_baseline.py \
  --model_type baseline \
  --model_name bert-base-uncased \
  --batch_size 16 \
  --num_epochs 3 \
  --learning_rate 2e-5 \
  --loss_type bce \
  --dropout 0.1 \
  --max_length 128 \
  --train_labels_path data/intermediate/train_silver_labels.pkl \
  --output_dir models/baseline
```

### Option B: Config 기반

```bash
python3 scripts/train_with_config.py --config config/config.yaml
```

### 주요 파라미터 설명

| 파라미터 | 설명 | Default | 옵션 |
|---------|------|---------|------|
| `--model_type` | 모델 타입 (디렉토리 이름) | `baseline` | baseline, gcn, gat, focal_loss 등 |
| `--model_name` | BERT 모델 | `bert-base-uncased` | roberta-base, distilbert 등 |
| `--batch_size` | 배치 크기 | `16` | 8, 16, 32 (GPU 메모리에 따라) |
| `--num_epochs` | 에포크 수 | `3` | 3-5 |
| `--learning_rate` | 학습률 | `2e-5` | 1e-5 ~ 5e-5 |
| `--loss_type` | Loss 함수 | `bce` | bce, focal, asymmetric, kld |
| `--dropout` | Dropout 비율 | `0.1` | 0.1 ~ 0.3 |
| `--output_dir` | 출력 디렉토리 | `models/{model_type}` | 자동 생성 |

### 생성 파일
```
models/baseline/
├── checkpoint_epoch_1.pt
├── checkpoint_epoch_2.pt
├── checkpoint_epoch_3.pt
├── best_model.pt              # 최고 성능 모델
├── final_model.pt             # 마지막 에포크 모델
└── training_history.json      # 학습 히스토리
```

### 예상 결과
- Training loss: 0.3-0.5
- 소요 시간: 2-3시간 (V100 GPU 기준)

---

## 🔹 STEP 3: Self-Training (Optional)

### 명령어

```bash
python3 src/training/train_baseline.py \
  --model_type baseline_self_training \
  --use_self_training \
  --self_training_confidence 0.7 \
  --self_training_iterations 3 \
  --num_epochs 2 \
  --batch_size 16 \
  --loss_type bce \
  --output_dir models/baseline_self_training
```

### Self-Training 파라미터

| 파라미터 | 설명 | Default |
|---------|------|---------|
| `--use_self_training` | Self-training 활성화 | False |
| `--self_training_confidence` | Pseudo-label confidence threshold | 0.7 |
| `--self_training_iterations` | 최대 반복 횟수 | 3 |

### 동작 방식
1. Labeled data로 초기 학습
2. Test data에 대해 soft pseudo-label 생성 (confidence ≥ 0.7)
3. Labeled + Pseudo-labeled 데이터로 재학습 (KLD loss)
4. 2-3번 반복 또는 수렴 시 종료

### 생성 파일
```
models/baseline_self_training/
├── best_model.pt
├── final_model.pt
└── training_history.json
```

---

## 🔹 STEP 4: Inference (예측 생성)

### 명령어

```bash
python3 src/inference/predict.py \
  --model_path models/baseline/best_model.pt \
  --model_name baseline \
  --batch_size 32 \
  --threshold 0.3 \
  --device cuda
```

### 파라미터

| 파라미터 | 설명 | Default |
|---------|------|---------|
| `--model_path` | 학습된 모델 경로 | 필수 |
| `--model_name` | 모델 이름 (출력 파일명용) | baseline |
| `--batch_size` | 배치 크기 | 32 |
| `--threshold` | 예측 threshold | 0.3 |
| `--device` | 디바이스 | cuda/mps/cpu |

### 생성 파일
```
results/predictions/
├── baseline_20251213_150430.pkl       # Pickle 형식
└── baseline_20251213_150430.csv       # CSV 형식 (제출용)
```

### CSV 형식
```
# results/predictions/baseline_YYYYMMDD_HHMMSS.csv
# 각 줄: space-separated class IDs
5 12 103 245
8 25 67 201 350
1 45 200
...
```

---

## 🔹 STEP 5: Submission 파일 생성

### 명령어

```bash
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_20251213_150430.csv \
  --output results/submissions/20252R0136_baseline.csv \
  --student_id 20252R0136
```

### 또는 PKL에서 직접 생성

```bash
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_20251213_150430.pkl \
  --output results/submissions/20252R0136_baseline.csv
```

### 생성 파일
```
results/submissions/
└── 20252R0136_baseline.csv    # Kaggle 제출 파일
```

---

## 최종 디렉토리 구조

```
data304_final/
├── METHODOLOGY.md                      # 방법론 문서
├── PIPELINE.md                         # 이 파일
├── README.md
├── requirements.txt
├── run.sh
│
├── config/
│   └── config.yaml                     # 전체 설정 파일
│
├── data/
│   ├── raw/
│   │   └── Amazon_products/            # 원본 데이터
│   │       ├── classes.txt
│   │       ├── class_hierarchy.txt
│   │       ├── class_related_keywords.txt
│   │       ├── train/
│   │       │   └── train_corpus.txt
│   │       └── test/
│   │           └── test_corpus.txt
│   │
│   ├── intermediate/
│   │   ├── train_silver_labels.pkl     # ✅ STEP 1 생성
│   │   └── test_silver_labels.pkl      # ✅ STEP 1 생성
│   │
│   └── output/                          # (미사용)
│
├── models/
│   ├── baseline/                        # ✅ STEP 2 생성
│   │   ├── checkpoint_epoch_1.pt
│   │   ├── checkpoint_epoch_2.pt
│   │   ├── checkpoint_epoch_3.pt
│   │   ├── best_model.pt               # 최고 성능
│   │   ├── final_model.pt
│   │   └── training_history.json
│   │
│   ├── baseline_self_training/          # ✅ STEP 3 생성 (optional)
│   │   ├── best_model.pt
│   │   ├── final_model.pt
│   │   └── training_history.json
│   │
│   ├── focal_loss/                      # 추가 실험용
│   ├── asymmetric_loss/                 # 추가 실험용
│   ├── gcn/                             # 추가 실험용
│   └── gat/                             # 추가 실험용
│
├── results/
│   ├── predictions/                     # ✅ STEP 4 생성
│   │   ├── baseline_20251213_150430.pkl
│   │   ├── baseline_20251213_150430.csv
│   │   ├── baseline_self_training_20251213_163020.pkl
│   │   └── baseline_self_training_20251213_163020.csv
│   │
│   ├── submissions/                     # ✅ STEP 5 생성
│   │   ├── 20252R0136_baseline.csv      # 최종 제출 파일
│   │   └── 20252R0136_self_training.csv
│   │
│   └── images/                          # 시각화 결과
│       ├── eda/
│       ├── baseline/
│       ├── ablation/
│       └── case_study/
│
├── logs/                                # 학습 로그
│   ├── training_20251213_150430.log
│   └── silver_labels_20251213_140230.log
│
├── notebooks/
│   ├── EDA.ipynb
│   ├── Visualization.ipynb
│   ├── Ablation_Analysis.ipynb
│   └── CaseStudy.ipynb
│
├── scripts/
│   ├── generate_labels.py               # Silver label 생성 (config 기반)
│   ├── train_with_config.py             # 학습 (config 기반)
│   ├── generate_submission.py           # 제출 파일 생성
│   └── test/
│       ├── test_silver_labels.py
│       ├── test_training_pipeline.py
│       ├── test_predict.py
│       └── test_submission.py
│
└── src/
    ├── data_preprocessing.py
    ├── dataset.py
    │
    ├── models/
    │   ├── encoder.py                   # BERT TextEncoder
    │   ├── classifier.py
    │   └── gnn_classifier.py
    │
    ├── training/
    │   ├── train_baseline.py            # ⭐ 메인 학습 스크립트
    │   ├── self_training.py             # ⭐ Self-training (soft labels + KLD)
    │   └── loss_functions.py            # BCE, Focal, Asymmetric, KLD
    │
    ├── inference/
    │   ├── predict.py                   # ⭐ 예측 생성
    │   └── dummy_baseline.py
    │
    ├── silver_labeling/
    │   ├── generate_silver_labels.py    # ⭐ Hybrid top-down labeling
    │   ├── graph_utils.py
    │   └── llm_keyword_expansion.py
    │
    └── utils/
        ├── metrics.py
        ├── logger.py
        ├── seed.py
        └── taxonomy_mapping.py
```

---

## 전체 파이프라인 실행 예제

### 1. 로컬 테스트

```bash
# Step 1: Silver labels 생성
python3 src/silver_labeling/generate_silver_labels.py

# Step 2: Quick test (100 samples, 1 epoch)
python3 scripts/test/test_training_pipeline.py

# Step 3: Small training (전체 데이터, 1 epoch for testing)
python3 src/training/train_baseline.py \
  --model_type baseline_test \
  --num_epochs 1 \
  --batch_size 8 \
  --output_dir models/baseline_test
```

### 2. AWS 전체 학습

```bash
# Step 1: Silver labels (이미 생성되어 있으면 스킵)
python3 scripts/generate_labels.py

# Step 2: Baseline training
python3 src/training/train_baseline.py \
  --model_type baseline \
  --batch_size 16 \
  --num_epochs 3 \
  --output_dir models/baseline

# Step 3: Prediction
python3 src/inference/predict.py \
  --model_path models/baseline/best_model.pt \
  --model_name baseline

# Step 4: Generate submission
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_YYYYMMDD_HHMMSS.csv \
  --output results/submissions/20252R0136_baseline.csv
```

### 3. Self-Training 실험

```bash
# Baseline with self-training
python3 src/training/train_baseline.py \
  --model_type baseline_self_training \
  --use_self_training \
  --self_training_confidence 0.7 \
  --self_training_iterations 3 \
  --num_epochs 2 \
  --batch_size 16

# Prediction
python3 src/inference/predict.py \
  --model_path models/baseline_self_training/best_model.pt \
  --model_name baseline_self_training

# Submission
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_self_training_YYYYMMDD_HHMMSS.csv \
  --output results/submissions/20252R0136_self_training.csv
```

---

## Ablation Studies (추가 실험)

### Focal Loss

```bash
python3 src/training/train_baseline.py \
  --model_type focal_loss \
  --loss_type focal \
  --batch_size 16 \
  --num_epochs 3
```

### Asymmetric Loss

```bash
python3 src/training/train_baseline.py \
  --model_type asymmetric_loss \
  --loss_type asymmetric \
  --batch_size 16 \
  --num_epochs 3
```

### Different BERT Models

```bash
# RoBERTa
python3 src/training/train_baseline.py \
  --model_type roberta_baseline \
  --model_name roberta-base \
  --batch_size 16 \
  --num_epochs 3

# DistilBERT (faster)
python3 src/training/train_baseline.py \
  --model_type distilbert_baseline \
  --model_name distilbert-base-uncased \
  --batch_size 32 \
  --num_epochs 3
```

---

## 예상 소요 시간 (AWS p3.2xlarge - V100 GPU)

| 단계 | 소요 시간 | 비고 |
|-----|---------|-----|
| **Silver Label 생성** | 3-5분 | 최초 1회 |
| **Baseline Training (3 epochs)** | 2-3시간 | batch_size=16 |
| **Self-Training (3 iterations)** | 4-6시간 | iteration당 1.5-2시간 |
| **Inference** | 10-15분 | batch_size=32 |
| **Total (baseline)** | ~3시간 | Silver + Train + Inference |
| **Total (self-training)** | ~6시간 | Silver + Self-Train + Inference |

---

## 체크리스트

### 실행 전 확인사항

- [ ] 가상환경 활성화: `source data304/bin/activate`
- [ ] 의존성 설치: `pip install -r requirements.txt`
- [ ] 데이터 존재 확인: `data/raw/Amazon_products/`
- [ ] Config 설정 확인: `config/config.yaml`
- [ ] GPU 사용 가능 확인: `nvidia-smi` (AWS)

### 실행 후 확인사항

- [ ] Silver labels 생성: `data/intermediate/train_silver_labels.pkl`
- [ ] 모델 저장: `models/{model_type}/best_model.pt`
- [ ] 학습 히스토리: `models/{model_type}/training_history.json`
- [ ] 예측 파일: `results/predictions/*.csv`
- [ ] 제출 파일: `results/submissions/20252R0136_*.csv`

### 제출 전 확인사항

- [ ] CSV 포맷 확인: 각 줄이 space-separated integers
- [ ] 라인 수 확인: 19,658 lines (test set size)
- [ ] 파일명 형식: `20252R0136_*.csv`
- [ ] 파일 크기: ~1-5MB

---

## 트러블슈팅

### Out of Memory

```bash
# 배치 크기 줄이기
--batch_size 8

# Gradient accumulation 사용
--gradient_accumulation_steps 2
```

### Slow Training

```bash
# DistilBERT 사용 (50% faster)
--model_name distilbert-base-uncased

# Mixed precision training (AWS GPU only)
# config.yaml에서 mixed_precision: true
```

### Poor Performance

```bash
# 더 많은 epoch
--num_epochs 5

# Self-training 사용
--use_self_training

# Different loss function
--loss_type focal  # or asymmetric
```

---

## 다음 단계

1. **EDA 노트북 실행**: `notebooks/EDA.ipynb`
2. **Ablation 분석**: `notebooks/Ablation_Analysis.ipynb`
3. **Case Study**: `notebooks/CaseStudy.ipynb`
4. **결과 시각화**: `notebooks/Visualization.ipynb`
5. **레포트 작성**: 결과 정리 및 분석

---

## 참고 문서

- **METHODOLOGY.md**: 전체 방법론 상세 설명
- **README.md**: 프로젝트 개요
- **config/config.yaml**: 설정 파일 주석
