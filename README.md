# Hierarchical Multi-Label Product Classification

Amazon 상품 리뷰의 계층적 다중 레이블 분류 프로젝트

## 프로젝트 구조

```
data304_final/
├── config/
│   └── config.yaml              # 학습 하이퍼파라미터 설정
├── data/
│   ├── raw/                     # 원본 데이터
│   ├── intermediate/            # 중간 생성 파일 (silver labels 등)
│   └── output/                  # 최종 결과물
├── src/
│   ├── data_preprocessing.py    # 데이터 로딩
│   ├── dataset.py               # PyTorch Dataset
│   ├── models/                  # 모델 정의
│   ├── training/                # 학습 관련 코드
│   ├── silver_labeling/         # Silver label 생성
│   └── utils/                   # 유틸리티 함수
├── scripts/                     # 실행 스크립트
│   ├── train_with_config.py     # Config 기반 학습
│   ├── train_sagemaker.sh       # AWS SageMaker 학습 스크립트
│   ├── test_training_pipeline.py # 파이프라인 테스트
│   └── check_silver_labels.py   # Silver label 통계 확인
├── notebooks/                   # 분석 노트북
└── models/                      # 학습된 모델 저장

```

## 설치

```bash
# 가상환경 생성
python3 -m venv data304
source data304/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

## 실행 방법

### 1. Silver Label 생성

```bash
python3 src/silver_labeling/generate_silver_labels.py
```

### 2. 파이프라인 테스트 (로컬)

```bash
python3 scripts/test_training_pipeline.py
```

### 3. 학습 (Config 사용)

```bash
python3 scripts/train_with_config.py --config config/config.yaml
```

### 4. AWS SageMaker에서 학습

```bash
./scripts/train_sagemaker.sh
```

## 주요 기능

- **Silver Label 생성**: 키워드 매칭 기반 자동 레이블링 (Coverage: 70%)
- **BERT 기반 인코더**: 사전학습된 언어모델 활용
- **다양한 Loss Functions**: BCE, Focal, Asymmetric, Hierarchical Loss
- **계층 구조 활용**: Taxonomy 정보 통합

## 구현 예정

- [ ] Self-training
- [ ] GNN 기반 분류
- [ ] Kaggle 제출 파이프라인

## 성능

- Silver Label Coverage: 70.03% (train), 70.13% (test)
- Class Coverage: 445/531 (83.8%)
- 평균 레이블 수: 3.25개/문서
