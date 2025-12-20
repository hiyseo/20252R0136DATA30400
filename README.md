# 🛍️ TaxoClass: Amazon 상품 계층적 분류 시스템

**과목:** DATA304  
**학번:** 2020320135

---

## 📋 프로젝트 개요

Amazon 상품을 **531개 계층적 클래스**로 자동 분류하는 Self-Training 기반 텍스트 분류 시스템입니다.

### 주요 특징
- **Silver Label 생성**: Sentence-BERT 기반 의미 유사도
- **계층 구조 활용**: Parent-Child 관계 반영
- **Self-Training**: Pseudo-labeling으로 성능 향상
- **완전 자동화**: 레이블 없이도 높은 정확도 달성

---

## 🚀 실행 방법

### 옵션 1: Google Colab에서 실행 (권장 ⭐)

1. **데이터 준비**
   - `Amazon_products` 폴더를 Google Drive에 업로드
   - 경로: `/content/drive/MyDrive/Amazon_products/`
   
2. **노트북 실행**
   - `TaxoClass_st_overall_reports.ipynb`를 Colab에 업로드
   - 순서대로 셀 실행 (모든 코드와 시각화 포함)

3. **주요 파일 구조 (Google Drive)**
   ```
   /content/drive/MyDrive/Amazon_products/
   ├── classes.txt                      # 531개 클래스 정보
   ├── class_hierarchy.txt              # 568개 계층 관계
   ├── class_related_keywords.txt       # 클래스별 키워드
   ├── train/
   │   └── train_corpus.txt            # 학습 문서
   └── test/
       └── test_corpus.txt             # 테스트 문서
   ```

---

### 옵션 2: 로컬 환경에서 실행

#### 1. 환경 설정

```bash
# 저장소 클론
git clone https://github.com/hiyseo/20252R0136DATA30400.git
cd 20252R0136DATA30400

# 가상환경 생성 및 활성화
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 패키지 설치
pip install -r requirements.txt
```

**시스템 요구사항:**
- Python 3.8+
- 16GB RAM (권장)
- GPU (선택, CUDA 지원 시 빠름)

#### 2. 데이터 준비

```bash
# 프로젝트 루트에 data 폴더 생성
mkdir -p data/raw/Amazon_products/train
mkdir -p data/raw/Amazon_products/test

# 데이터 파일 배치 (아래 경로에 복사)
# data/raw/Amazon_products/
# ├── classes.txt
# ├── class_hierarchy.txt
# ├── class_related_keywords.txt
# ├── train/train_corpus.txt
# └── test/test_corpus.txt
```

**데이터 획득 방법:**
- 교수님/TA에게 문의

#### 3. 노트북 수정 및 실행

**경로 수정:**
노트북의 "Step 2: 데이터 로드" 섹션에서 경로 변경

```python
# Google Colab 경로 (기존)
BASE_PATH = '/content/drive/MyDrive/Amazon_products'

# 로컬 경로로 변경
BASE_PATH = './data/raw/Amazon_products'
```

**Colab 전용 코드 제거/주석처리:**
```python
# Google Drive 마운트 셀 (주석처리 또는 스킵)
# from google.colab import drive
# drive.mount('/content/drive')
```

**Jupyter 실행:**
```bash
# Jupyter Lab 실행
jupyter lab

# 또는 Jupyter Notebook
jupyter notebook
```

브라우저에서 `TaxoClass_st_overall_reports.ipynb` 열고 순서대로 실행

---

## 📊 노트북 구조

노트북은 **완전히 독립적**으로 실행 가능하며, 다음 내용을 포함합니다:

1. **환경 설정** - 라이브러리 설치 및 import
2. **데이터 로드** - 클래스, 계층, 키워드, 문서
3. **EDA** - 데이터 분포 및 계층 구조 분석
4. **Silver Label 생성** - Sentence-BERT 기반 의사 레이블링
5. **모델 학습** - BCE Loss + Self-Training
6. **평가 및 시각화** - 성능 메트릭, Confusion Matrix, 케이스 스터디

---

## 📂 프로젝트 구조 (최소 버전)

```
data304_final/
├── README.md                             # 이 파일
├── TaxoClass_st_overall_reports.ipynb   # 실행 가능한 전체 노트북
├── requirements.txt                      # Python 패키지 목록
├── .gitignore                           # Git 제외 파일
└── data/                                # (로컬 전용, .gitignore에 포함)
    └── raw/
        └── Amazon_products/
            ├── classes.txt
            ├── class_hierarchy.txt
            ├── class_related_keywords.txt
            ├── train/train_corpus.txt
            └── test/test_corpus.txt
```

---

## 🔧 문제 해결 (Troubleshooting)

### 1. 패키지 설치 오류
```bash
# pip 업그레이드
pip install --upgrade pip

# 개별 설치
pip install torch sentence-transformers scikit-learn
```

### 2. CUDA/GPU 오류
```python
# CPU로 강제 실행 (노트북 상단에 추가)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
```

### 3. 메모리 부족
- Colab Pro 사용 또는 배치 크기 감소
- 로컬: 스왑 메모리 증가

### 4. 데이터 경로 오류
- 노트북의 `BASE_PATH` 변수 확인
- 파일 존재 여부 확인: `ls data/raw/Amazon_products/`

---

## 📈 주요 결과

- **Silver Label Accuracy**: 85-90%
- **Self-Training 개선**: +5-7% (3 iterations)
- **계층 구조 활용**: Parent-Child 일관성 향상

---

## 📝 라이선스

이 프로젝트는 DATA304 과제용으로 작성되었습니다.

---

## 📧 문의

- **학번**: 2020320135
- **GitHub**: [hiyseo/20252R0136DATA30400](https://github.com/hiyseo/20252R0136DATA30400)
