# Amazon 상품 계층 분류 (Hierarchical Product Classification)

**학번:** 20252R0136 | **과목:** DATA304

---

## 📋 개요

Amazon 상품을 531개 클래스로 자동 분류하는 시스템입니다.

**방법**: Silver Label 생성 → BCE 학습 → Self-Training (KLD)  
**특징**: 레이블 없이도 높은 성능 달성

---

## 📁 프로젝트 구조

```
data304_final/
├── config/                      # 설정 파일
│   └── config.yaml             # 통합 실험 설정
│
├── data/                        # 데이터 디렉토리
│   ├── raw/                    # 원본 데이터 (Amazon Products)
│   │   └── Amazon_products/    # Amazon 상품 데이터셋
│   │       ├── train/          # 학습 데이터 (corpus)
│   │       ├── test/           # 테스트 데이터 (corpus)
│   │       ├── classes.txt     # 531개 클래스 목록
│   │       ├── class_hierarchy.txt  # 계층 구조 (parent-child)
│   │       └── class_related_keywords.txt  # 클래스별 키워드
│   ├── intermediate/           # 중간 처리 결과
│   │   ├── train_silver_labels.pkl  # 학습 데이터 Silver label
│   │   └── test_silver_labels.pkl   # 테스트 데이터 Silver label
│   └── output/                 # 최종 출력 (예측 결과)
│
├── src/                         # 소스 코드
│   ├── data_preprocessing.py   # 데이터 전처리
│   ├── models/                 # 모델 정의
│   │   ├── classifier.py       # BERT 기반 분류기
│   │   ├── encoder.py          # BERT 인코더
│   │   └── gnn_classifier.py   # GNN 모델 (Graph Neural Network)
│   ├── silver_labeling/        # Silver label 생성
│   │   ├── generate_silver_labels.py  # 메인 실행 파일
│   │   ├── graph_utils.py      # 계층 그래프 처리
│   │   └── llm_keyword_expansion.py  # LLM 키워드 확장 (선택)
│   ├── training/               # 학습 로직
│   │   ├── train_baseline.py   # 2단계 학습 (BCE → Self-Training)
│   │   ├── self_training.py    # Self-training 구현
│   │   └── loss_functions.py   # 손실 함수 (BCE, KLD, Focal)
│   ├── inference/              # 예측 생성
│   │   ├── predict.py          # 모델 예측
│   │   └── dummy_baseline.py   # 더미 베이스라인
│   └── utils/                  # 유틸리티
│       ├── metrics.py          # 평가 지표 (F1, Precision, Recall)
│       ├── taxonomy_mapping.py # 계층 매핑
│       ├── logger.py           # 로깅
│       └── seed.py             # 랜덤 시드 고정
│
├── scripts/                     # 실행 스크립트
│   ├── generate_labels.py      # Silver label 생성 실행
│   ├── train_with_config.py    # Config 기반 학습 실행
│   └── generate_submission.py  # 제출 파일 생성
│
├── notebooks/                   # Jupyter Notebook 분석
│   ├── EDA.ipynb               # 데이터 탐색 및 시각화
│   ├── Ablation_Analysis.ipynb # 실험 결과 비교
│   └── CaseStudy.ipynb         # 예측 오류 분석
│
├── docs/                        # 문서
│   ├── CONFIG.md               # Config 상세 설명
│   ├── PIPELINE.md             # 파이프라인 실행 가이드
│   └── METHODOLOGY.md          # 방법론 및 수식
│
├── models/                      # 학습된 모델 (자동 생성)
│   └── {model_type}/           # 실험별 폴더
│       ├── best_model.pt       # 최종 모델
│       ├── training_history.json  # 학습 기록
│       └── checkpoint_*.pt     # 중간 체크포인트
│
├── results/                     # 결과 파일 (자동 생성)
│   ├── predictions/            # 예측 결과 (pkl, csv)
│   ├── submissions/            # 제출 파일
│   └── images/                 # 시각화 이미지
│
├── logs/                        # 로그 파일 (자동 생성)
│
├── run.sh                       # 전체 파이프라인 자동 실행 스크립트
├── requirements.txt             # Python 패키지 의존성
└── README.md                    # 프로젝트 설명서
```

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

## ⚙️ Config 설정

`config/config.yaml` 파일 수정으로 실험 설정:

### 주요 옵션

```yaml
# 모델 설정
model:
  model_name: "bert-base-uncased"  # 사전학습 모델
    # 옵션: "bert-base-uncased", "roberta-base", "distilbert-base-uncased"
  model_type: "baseline"  # 실험 이름 (출력 폴더명으로 사용)
    # 예시: "baseline", "focal_loss", "gcn", "gat", "no_self_training"
  dropout: 0.1  # Dropout 비율 (과적합 방지)

# 학습 설정
training:
  batch_size: 16  # 배치 크기 (GPU 메모리에 따라 조정)
    # CPU: 4-8, GPU (8GB): 16, GPU (16GB+): 32
  num_epochs: 2  # Stage 1 초기 학습 에포크
  learning_rate: 2.0e-5  # 학습률
  loss_type: "bce"  # 손실 함수
    # 옵션: "bce" (Binary Cross Entropy), "focal" (Focal Loss)
  
  # Focal Loss 설정 (loss_type: "focal"일 때)
  focal_alpha: 0.25  # 클래스 불균형 보정
  focal_gamma: 2.0   # 쉬운 샘플 가중치 감소

# Self-Training 설정
self_training:
  enabled: true  # Self-training 활성화
    # true: BCE (Stage 1) → KLD (Stage 2)
    # false: BCE만 사용
  confidence_threshold: 0.7  # Pseudo-label 신뢰도 임계값
    # 높을수록 엄격 (0.6-0.9 권장)
  max_iterations: 3  # Self-training 반복 횟수

# 데이터 설정
data:
  max_length: 128  # 텍스트 최대 토큰 길이
    # 메모리 부족 시: 64, 긴 텍스트: 256
  num_workers: 4  # 데이터 로딩 병렬 처리 수

# 환경 설정
misc:
  device: "auto"  # 디바이스 자동 선택
    # 옵션: "auto", "cpu", "cuda" (NVIDIA GPU), "mps" (Apple Silicon)
  seed: 42  # 재현성을 위한 랜덤 시드
  mixed_precision: true  # 혼합 정밀도 학습 (GPU 속도 향상)
```

### 실험 시나리오별 설정

**빠른 테스트 (5-10분)**
```yaml
training:
  num_epochs: 1
  batch_size: 8
self_training:
  enabled: false
```

**메모리 부족 시**
```yaml
data:
  max_length: 64
  batch_size: 4
misc:
  mixed_precision: true
```

**고성능 GPU (긴 학습)**
```yaml
training:
  num_epochs: 5
  batch_size: 32
data:
  max_length: 256
self_training:
  max_iterations: 5
```

**Focal Loss 실험**
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

**비용**: 100-200 클래스 확장 시 $2-5  
**선택사항**: 없어도 정상 작동 (기본 키워드로도 충분)

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