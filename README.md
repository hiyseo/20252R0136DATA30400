# 🛍️ Amazon Hierarchical Product Classification

**학번:** 2020320135 | **과목:** DATA304

---

## 📋 개요

Amazon 상품을 531개 클래스로 자동 분류하는 시스템입니다.

**방법**: Silver Label 생성 → BCE 학습 → Self-Training (KLD)  
**특징**: 레이블 없이도 높은 성능 달성

---

## 📁 프로젝트 구조

**핵심 디렉토리:**
- `config/`: 실험 설정 (단일 YAML 파일)
- `data/`: 원본 → 중간 처리 결과
  - `raw/Amazon_products/`: 원본 데이터 (train/test corpus)
  - `intermediate/`: 전처리 결과 (silver labels)
  - `models/`: 학습된 모델 저장 (.pt, .json)
- `scripts/`: 실행 진입점 (config 기반)
- `results/`: 모든 실험 결과 저장
  - `training/`: 학습 시각화 (loss curves)
  - `evaluation/`: 평가 메트릭 및 시각화
  - `images/`: Jupyter 노트북 결과 저장 경로
  - `predictions/`: 예측 결과 (pkl)
  - `submissions/`: 제출 파일 (csv)
- `src/`: 모든 소스 코드 (모델, 학습, 추론, 평가)

---

## 🚀 빠른 시작

### 1. 환경 설정

```bash
# 가상환경 생성 및 활성화
python3 -m venv data304
source data304/bin/activate

# 패키지 설치
pip install -r requirements.txt
```

**필요**: Python 3.10+, 16GB RAM, 10GB 저장공간

### 2. 전체 실행 (한 번에)

```bash
# 실행 권한 부여 (최초 1회)
chmod +x run.sh

# 전체 파이프라인 자동 실행
./run.sh
```

**소요 시간**: CPU 12-16시간, GPU 3-4시간

### 3. 단계별 실행 (선택)

```bash
# 1. 데이터 전처리
python3 src/data_preprocessing.py

# 2. Silver Label 생성 (30-45분)
python3 scripts/generate_labels.py

# 3. 모델 학습 (Stage 1: BCE → Stage 2: Self-Training)
python3 scripts/train_with_config.py

# 4. 예측 생성
python3 src/inference/predict.py \
  --model_path data/output/models/baseline/best_model.pt \
  --model_name baseline

# 5. 제출 파일 생성
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_*.csv \
  --output results/submissions/2020320135_baseline.csv
```

**학습 중 자동 생성**:
- `data/output/models/baseline/best_model.pt` - 최종 모델
- `data/output/models/baseline/training_history.json` - 학습 기록
- `results/training/baseline/*.png` - 학습 시각화 (loss curves)

**최종 출력**: `results/submissions/2020320135_baseline.csv` (제출용)

---

### 4. 모델 평가 (선택)

```bash
# 단독 실행
python3 src/evaluation/evaluate_model.py \
  --model_path data/models/baseline/best_model.pt \
  --model_type baseline \
  --save_predictions

# 또는 run.sh로 실행 (Step 3.5)
./run.sh --step 3.5
```

**평가 데이터**: Test set (19,658 samples) with silver labels (pseudo ground truth)  
**주의**: Silver label을 정답으로 사용하므로 실제 성능과 다를 수 있습니다.

**출력 위치**: `results/evaluation/{model_type}/`

**평가 메트릭** (Multi-Label Classification):
- **Micro F1/Precision/Recall**: 전체 예측의 정확도 (클래스 빈도 가중)
- **Macro F1/Precision/Recall**: 클래스별 평균 (클래스 불균형 무시)
- **Samples F1**: 샘플별 F1 평균 (문서 단위 성능)
- **Top-k Accuracy**: 상위 k개 예측 중 정답 포함 비율 (k=3, 5)
- **Exact Match Ratio**: 모든 레이블이 정확히 일치하는 비율

**생성 파일** (6개 시각화):
1. `eval_{model_name}_metrics.json` - 상세 메트릭 (JSON)
2. `eval_{model_name}_confidence_distribution.png` - 예측 신뢰도 분포 (positive/negative)
3. `eval_{model_name}_labels_per_sample_distribution.png` - 샘플당 레이블 수 분포
4. `eval_{model_name}_metrics.png` - 전체 메트릭 막대 그래프
5. `eval_{model_name}_f1_precision_recall.png` - F1/Precision/Recall 비교
6. `eval_{model_name}_topk_accuracy.png` - Top-3/Top-5/Exact Match 정확도
7. `eval_{model_name}_per_class_performance.png` - 클래스별 성능 (상위/하위 10개)

**실전 성능**: Kaggle 제출 후 실제 성능 확인 필요

---

## 🖥️ 실행 환경

### 로컬 (CPU/GPU)

```yaml
# config/config.yaml
misc:
  device: "auto"  # 또는 "cpu", "cuda", "mps"
  
training:
  batch_size: 16  # CPU는 8, GPU는 32
```

**예상 시간**: CPU 12-16시간, GPU 3-6시간

---

### 실험 시나리오별 설정 (Model Type)

#### 1. Baseline (2-Stage Training)
```yaml
model:
  model_type: "baseline"
training:
  loss_type: "bce"
  num_epochs: 2
self_training:
  enabled: true  # BCE → Self-Training (KLD)
  confidence_threshold: 0.7
  max_iterations: 3
```

#### 2. Focal Loss (클래스 불균형 해결)
```yaml
model:
  model_type: "focal_loss"
training:
  loss_type: "focal"
  focal_alpha: 0.25
  focal_gamma: 2.0
  num_epochs: 5
self_training:
  enabled: false
```

#### 3. Self-Training 없이 (BCE만)
```yaml
model:
  model_type: "no_self_training"
training:
  loss_type: "bce"
  num_epochs: 5
self_training:
  enabled: false
```

#### 4. 빠른 테스트
```yaml
model:
  model_type: "quick_test"
training:
  num_epochs: 1
  batch_size: 8
self_training:
  enabled: false
```

**상세 설명**: `docs/CONFIG.md` 참조

---

## 🔑 LLM API 설정 (선택사항)

키워드 확장을 위한 OpenAI API 설정 (선택):

```bash
# 1. API 키 발급: https://platform.openai.com/
# 2. 환경 변수 설정
echo "OPENAI_API_KEY=sk-proj-..." > .env

# 3. Config 활성화
# config/config.yaml
silver_labeling:
  llm_expansion:
    enabled: true
    model: "gpt-4o-mini"
    max_calls: 1000  # 비용 제한

# 4. 실행
python3 src/silver_labeling/llm_keyword_expansion.py
```

**상세 설명**: `docs/LLM_KEYWORD_EXPANSION.md` 참조

---

## 📊 방법론

### 3단계 파이프라인

1. **Silver Label 생성**: 키워드 매칭(30%) + 임베딩 유사도(70%)
2. **Stage 1 (BCE)**: Hard label로 초기 학습 (2 epochs)
3. **Stage 2 (KLD)**: Soft pseudo-label로 self-training (3 iterations)

---

## 🔬 실험 및 분석

Ablation study를 위한 Jupyter Notebook 제공:

```bash
jupyter notebook notebooks/Ablation_Analysis.ipynb  # 실험 비교
jupyter notebook notebooks/CaseStudy.ipynb          # 예측 분석
jupyter notebook notebooks/EDA.ipynb                # 데이터 탐색
```

**실험 예시**:
- Self-training 효과: `self_training.enabled: false`
- 임계값 변경: `confidence_threshold: 0.8`
- Loss 함수 비교: `loss_type: "focal"`

---