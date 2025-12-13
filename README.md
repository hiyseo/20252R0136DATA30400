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
- `src/`: 모든 소스 코드 (모델, 학습, 추론)
- `config/`: 실험 설정 (단일 YAML 파일)
- `data/`: 원본 → 중간 → 최종 데이터 흐름
- `scripts/`: 실행 진입점 (config 기반)
- `models/`: 학습된 모델 저장
- `results/`: 예측 및 제출 파일

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
  --model_path models/baseline/best_model.pt \
  --model_name baseline

# 5. 제출 파일 생성
python3 scripts/generate_submission.py \
  --predictions results/predictions/baseline_*.csv \
  --output results/submissions/20252R0136_baseline.csv
```

**최종 출력**: `results/submissions/20252R0136_baseline.csv` (제출용)

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