# LLM Keyword Expansion Guide

## Overview
LLM keyword expansion은 키워드가 부족한 클래스에 대해 추가 키워드를 자동 생성하여 silver label coverage를 향상시킵니다.

## Setup

### 1. Install Dependencies
```bash
pip install openai>=1.0.0
```

### 2. Set API Key
```bash
# OpenAI API Key 설정
export OPENAI_API_KEY="your-api-key-here"

# 또는 .env 파일에 저장
echo "OPENAI_API_KEY=your-api-key" > .env
```

### 3. Enable in Config
```yaml
# config/config.yaml
silver_labeling:
  llm_expansion:
    enabled: true
    model: "gpt-4o-mini"  # 또는 "gpt-3.5-turbo"
    max_calls: 1000
    min_keywords: 5  # 5개 미만 키워드인 클래스 우선 확장
    priority_limit: 100  # 최대 100개 클래스 확장
```

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
