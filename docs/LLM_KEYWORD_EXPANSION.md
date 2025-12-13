# LLM Keyword Expansion Guide

## Overview

LLM keyword expansion은 키워드가 부족한 클래스에 대해 추가 키워드를 자동 생성하여 silver label coverage를 향상시킵니다.

### 주요 기능

1. **자동 키워드 생성**: OpenAI GPT 모델을 사용하여 클래스별 관련 키워드 확장
2. **우선순위 기반 확장**: 키워드가 적은 클래스 우선 처리
3. **비용 제어**: 최대 API 호출 횟수 제한 (기본 1,000회)
4. **완전한 로깅**: 모든 API 호출 입출력 자동 저장 (검증 가능)
5. **세션 관리**: 중단 후 재개 시 누적 호출 횟수 유지

### 사용 시나리오

- 키워드가 5개 미만인 클래스 확장 (약 100-200개 클래스)
- Silver label coverage 향상 (70% → 80%+)
- 특정 도메인 키워드 강화

### 예상 비용

- **모델**: gpt-4o-mini ($0.15/1M input tokens, $0.60/1M output tokens)
- **예상**: 100-200 클래스 확장 시 약 $2-5
- **제한**: max_calls로 비용 상한 설정 가능

## Setup

### 1. Install Dependencies
```bash
pip install openai>=1.0.0 python-dotenv
```

### 2. OpenAI API Key 발급

1. **OpenAI Platform 접속**: https://platform.openai.com/
2. **로그인** 후 우측 상단 프로필 → **API Keys** 클릭
3. **Create new secret key** 클릭
4. 키 이름 입력 (예: "data304_project") → **Create secret key**
5. 생성된 키 복사 (⚠️ 한 번만 표시됨 - 즉시 저장 필요)

### 3. API Key 설정

**방법 1: 환경 변수 (추천)**
```bash
# macOS/Linux
export OPENAI_API_KEY="sk-proj-..."

# 영구 설정 (bash)
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.bashrc
source ~/.bashrc

# 영구 설정 (zsh)
echo 'export OPENAI_API_KEY="sk-proj-..."' >> ~/.zshrc
source ~/.zshrc
```

**방법 2: .env 파일 (보안)**
```bash
# 프로젝트 루트에 .env 파일 생성
echo "OPENAI_API_KEY=sk-proj-..." > .env

# .gitignore에 추가되어 있는지 확인 (이미 추가됨)
grep ".env" .gitignore
```

**방법 3: Config에 직접 (비권장)**
```yaml
# ⚠️ Git에 커밋하지 마세요!
silver_labeling:
  llm_expansion:
    api_key: "sk-proj-..."  # 비권장
```

### 4. Config 활성화

```yaml
# config/config.yaml
silver_labeling:
  llm_expansion:
    enabled: true  # LLM 키워드 확장 활성화
    model: "gpt-4o-mini"  # 추천 (저렴하고 빠름)
      # 옵션: "gpt-3.5-turbo" (더 저렴), "gpt-4" (더 정확, 비쌈)
    max_calls: 1000  # 최대 API 호출 횟수 (비용 제한)
    min_keywords: 5  # 키워드 5개 미만인 클래스만 확장
    priority_limit: 100  # 최대 확장 클래스 수
```

**파라미터 설명:**
- `enabled`: true로 설정 시 키워드 확장 활성화
- `model`: 사용할 GPT 모델 (gpt-4o-mini 추천)
- `max_calls`: 누적 API 호출 제한 (1,000회 = 약 $3-5)
- `min_keywords`: 이 개수 미만의 키워드를 가진 클래스 우선 확장
- `priority_limit`: 한 번에 확장할 최대 클래스 수

## Usage

### Option 1: Standalone Script
```bash
# 키워드 확장 실행
python3 src/silver_labeling/llm_keyword_expansion.py
```

### Option 2: Programmatic Usage
```python
from src.silver_labeling.llm_keyword_expansion import LLMKeywordExpander, prioritize_classes_by_coverage
from src.data_preprocessing import DataLoader

# Load data
data_loader = DataLoader()
data_loader.load_all()

# Initialize expander
expander = LLMKeywordExpander(
    model="gpt-4o-mini",
    log_dir="logs/llm_calls",
    max_calls=1000
)

# Prioritize classes with few keywords
priority_classes = prioritize_classes_by_coverage(
    data_loader.class_keywords,
    min_keywords=5
)

# Expand keywords
expanded_keywords = expander.expand_all_keywords(
    data_loader.class_keywords,
    priority_classes=priority_classes[:100]
)

# Save expanded keywords
import json
with open('data/intermediate/expanded_keywords.json', 'w') as f:
    json.dump(expanded_keywords, f, indent=2)
```

### Option 3: Integrate with Silver Label Generation
```python
from src.silver_labeling.generate_silver_labels import SilverLabelGenerator
from src.silver_labeling.llm_keyword_expansion import LLMKeywordExpander

# Expand keywords first
expander = LLMKeywordExpander(model="gpt-4o-mini", max_calls=1000)
expanded_keywords = expander.expand_all_keywords(data_loader.class_keywords)

# Use expanded keywords for silver labeling
generator = SilverLabelGenerator(
    class_keywords=expanded_keywords,  # Use expanded keywords
    class_to_id=data_loader.class_to_id,
    id_to_class=data_loader.id_to_class
)
```

## Logging

All LLM API calls are automatically logged for verification:

### Log Structure
```
logs/llm_calls/
├── call_history.json           # Total call count tracker
├── call_0001.json              # Individual call log #1
├── call_0002.json              # Individual call log #2
├── ...
├── session_20241212_143000.json  # Session summary
└── expanded_keywords_20241212_143000.json  # Expanded keywords result
```

### Log Contents

**Individual Call Log (`call_XXXX.json`):**
```json
{
  "call_id": 1,
  "session_id": "20241212_143000",
  "timestamp": "2024-12-12T14:30:05.123456",
  "model": "gpt-4o-mini",
  "class_name": "Gaming Laptops",
  "prompt": "You are helping to expand keywords...",
  "response": "high-performance, RTX, gaming PC, esports, ..."
}
```

**Session Summary (`session_YYYYMMDD_HHMMSS.json`):**
```json
{
  "session_id": "20241212_143000",
  "timestamp": "2024-12-12T14:35:00.000000",
  "model": "gpt-4o-mini",
  "total_calls": 95,
  "cumulative_calls": 95,
  "max_calls": 1000,
  "classes_expanded": 95,
  "calls": [...]
}
```

## Usage Policy Compliance

### Maximum Calls: 1,000
- Script automatically stops at 1,000 calls
- Call count persists across sessions in `call_history.json`
- Check remaining calls: `cat logs/llm_calls/call_history.json`

### Verification Files
모든 API 호출의 입력/출력이 개별 파일로 저장되어 과제 제출 시 검증 가능:

```bash
# 제출할 파일들
logs/llm_calls/call_*.json           # 모든 개별 호출 로그
logs/llm_calls/session_*.json        # 세션 요약
logs/llm_calls/call_history.json     # 전체 호출 카운트
```

## Cost Estimation

**GPT-4o mini pricing:**
- Input: $0.150 / 1M tokens
- Output: $0.600 / 1M tokens

**Estimated cost for 1,000 calls:**
- ~150 tokens/call (input) = 150K tokens = $0.0225
- ~50 tokens/call (output) = 50K tokens = $0.03
- **Total: ~$0.05 - $0.10** (well under $1 limit)

## Example Output

**Before:**
```
"Gaming Laptops": ["laptop", "gaming", "computer"]
```

**After LLM Expansion:**
```
"Gaming Laptops": [
  "laptop", "gaming", "computer",           # Original
  "high-performance", "RTX", "GPU",         # LLM generated
  "esports", "portable gaming", "gaming PC",
  "notebook", "gaming notebook", "mobile workstation"
]
```

## Tips

1. **Start with priority classes:** Classes with < 5 keywords benefit most
2. **Monitor API usage:** Check `call_history.json` frequently
3. **Save expanded keywords:** Store results before running silver label generation
4. **Batch processing:** Expand keywords once, use multiple times

## Troubleshooting

**No API client available:**
```bash
pip install openai
export OPENAI_API_KEY="your-key"
```

**Max calls reached:**
- Check `logs/llm_calls/call_history.json`
- Delete it to reset (use carefully!)

**Rate limiting:**
- Built-in 0.1s delay between calls
- Adjust in code if needed: `time.sleep(0.5)`
