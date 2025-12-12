"""
LLM-based keyword expansion for improving silver label coverage.
Uses GPT-4o mini or other LLMs to generate related keywords for each class.

Usage Policy:
- Maximum 1,000 API calls total (~$1)
- All prompts and outputs are logged for verification
"""

import json
import os
from typing import List, Dict, Optional
from datetime import datetime
from pathlib import Path
import time


class LLMKeywordExpander:
    """Expand class keywords using LLM."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 model: str = "gpt-4o-mini",
                 log_dir: str = "logs/llm_calls",
                 max_calls: int = 1000):
        """
        Initialize LLM keyword expander.
        
        Args:
            api_key: OpenAI API key (or None to use env variable)
            model: Model name (gpt-4o-mini, gpt-3.5-turbo, etc.)
            log_dir: Directory to save all LLM call logs
            max_calls: Maximum number of API calls allowed
        """
        self.model = model
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.max_calls = max_calls
        
        # Initialize API client
        if api_key is None:
            api_key = os.getenv("OPENAI_API_KEY")
        
        if api_key:
            try:
                from openai import OpenAI
                self.client = OpenAI(api_key=api_key)
            except ImportError:
                print("Warning: openai package not installed. Install with: pip install openai")
                self.client = None
        else:
            print("Warning: No API key provided. Set OPENAI_API_KEY environment variable.")
            self.client = None
        
        # Load call history
        self.call_history_file = self.log_dir / "call_history.json"
        self.call_count = self._load_call_count()
        
        # Session log
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_log = []
    
    def _load_call_count(self) -> int:
        """Load total API call count from history."""
        if self.call_history_file.exists():
            with open(self.call_history_file, 'r') as f:
                history = json.load(f)
                return history.get('total_calls', 0)
        return 0
    
    def _save_call_log(self, prompt: str, response: str, class_name: str):
        """Save individual call log."""
        self.call_count += 1
        
        call_log = {
            'call_id': self.call_count,
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'class_name': class_name,
            'prompt': prompt,
            'response': response
        }
        
        # Save to session log
        self.session_log.append(call_log)
        
        # Save individual call file
        call_file = self.log_dir / f"call_{self.call_count:04d}.json"
        with open(call_file, 'w', encoding='utf-8') as f:
            json.dump(call_log, f, indent=2, ensure_ascii=False)
        
        # Update call history
        history = {
            'total_calls': self.call_count,
            'last_updated': datetime.now().isoformat(),
            'last_session': self.session_id
        }
        with open(self.call_history_file, 'w') as f:
            json.dump(history, f, indent=2)
        
        print(f"  API Call {self.call_count}/{self.max_calls} logged")
    
    def _create_expansion_prompt(self, class_name: str, original_keywords: List[str]) -> str:
        """Create prompt for keyword expansion."""
        prompt = f"""You are helping to expand keywords for a product classification task.

Class Name: "{class_name}"
Original Keywords: {', '.join(original_keywords[:10])}

Task: Generate 10 additional related keywords, synonyms, or phrases that would help identify products belonging to this class.

Requirements:
- Focus on words that actually appear in product descriptions
- Include common abbreviations and alternative terms
- Avoid overly generic words
- Consider both formal and informal terminology

Return ONLY a comma-separated list of 10 keywords, nothing else.

Example output format: keyword1, keyword2, keyword3, ..."""
        
        return prompt
    
    def expand_keywords(self, class_name: str, original_keywords: List[str]) -> List[str]:
        """
        Expand keywords for a single class using LLM.
        
        Args:
            class_name: Name of the class
            original_keywords: Original keywords from dataset
            
        Returns:
            List of expanded keywords (original + new)
        """
        if self.client is None:
            print(f"  Skipping {class_name}: No API client available")
            return original_keywords
        
        if self.call_count >= self.max_calls:
            print(f"  Skipping {class_name}: Max calls ({self.max_calls}) reached")
            return original_keywords
        
        try:
            # Create prompt
            prompt = self._create_expansion_prompt(class_name, original_keywords)
            
            # Call LLM
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that generates relevant keywords for product classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=200
            )
            
            # Extract response
            response_text = response.choices[0].message.content.strip()
            
            # Save log
            self._save_call_log(prompt, response_text, class_name)
            
            # Parse new keywords
            new_keywords = [kw.strip() for kw in response_text.split(',')]
            new_keywords = [kw for kw in new_keywords if kw and len(kw) > 1]
            
            # Combine with original
            expanded = list(original_keywords) + new_keywords
            
            print(f"  ✓ {class_name}: {len(original_keywords)} → {len(expanded)} keywords")
            
            # Rate limiting
            time.sleep(0.1)
            
            return expanded
            
        except Exception as e:
            print(f"  ✗ Error expanding {class_name}: {e}")
            return original_keywords
    
    def expand_all_keywords(self, 
                           class_keywords: Dict[str, List[str]],
                           priority_classes: Optional[List[str]] = None) -> Dict[str, List[str]]:
        """
        Expand keywords for multiple classes.
        
        Args:
            class_keywords: Dictionary of class_name -> keywords
            priority_classes: Optional list of classes to expand first
            
        Returns:
            Dictionary with expanded keywords
        """
        expanded_keywords = {}
        
        # Determine order
        if priority_classes:
            # Expand priority classes first
            classes_to_expand = priority_classes + [c for c in class_keywords if c not in priority_classes]
        else:
            classes_to_expand = list(class_keywords.keys())
        
        print(f"\n=== Starting LLM Keyword Expansion ===")
        print(f"Model: {self.model}")
        print(f"Classes to expand: {len(classes_to_expand)}")
        print(f"Current API calls: {self.call_count}/{self.max_calls}")
        print(f"Session ID: {self.session_id}")
        print("="*60)
        
        for i, class_name in enumerate(classes_to_expand, 1):
            if self.call_count >= self.max_calls:
                print(f"\n⚠ Max API calls reached. Stopping expansion.")
                # Copy remaining classes as-is
                for remaining_class in classes_to_expand[i-1:]:
                    expanded_keywords[remaining_class] = class_keywords.get(remaining_class, [])
                break
            
            print(f"\n[{i}/{len(classes_to_expand)}] Expanding: {class_name}")
            original = class_keywords.get(class_name, [])
            expanded = self.expand_keywords(class_name, original)
            expanded_keywords[class_name] = expanded
        
        # Save session summary
        self._save_session_summary(expanded_keywords)
        
        print(f"\n=== Expansion Complete ===")
        print(f"Total API calls used: {self.call_count}/{self.max_calls}")
        print(f"Session logs saved to: {self.log_dir}")
        
        return expanded_keywords
    
    def _save_session_summary(self, expanded_keywords: Dict[str, List[str]]):
        """Save session summary."""
        summary = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'model': self.model,
            'total_calls': len(self.session_log),
            'cumulative_calls': self.call_count,
            'max_calls': self.max_calls,
            'classes_expanded': len(expanded_keywords),
            'calls': self.session_log
        }
        
        summary_file = self.log_dir / f"session_{self.session_id}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        # Save expanded keywords
        keywords_file = self.log_dir / f"expanded_keywords_{self.session_id}.json"
        with open(keywords_file, 'w', encoding='utf-8') as f:
            json.dump(expanded_keywords, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Session summary saved: {summary_file}")
        print(f"✓ Expanded keywords saved: {keywords_file}")


def prioritize_classes_by_coverage(class_keywords: Dict[str, List[str]], 
                                   min_keywords: int = 3) -> List[str]:
    """
    Prioritize classes with few keywords for expansion.
    
    Args:
        class_keywords: Dictionary of class_name -> keywords
        min_keywords: Minimum number of keywords to consider low coverage
        
    Returns:
        List of class names sorted by priority (lowest keyword count first)
    """
    classes_with_counts = [(name, len(keywords)) for name, keywords in class_keywords.items()]
    
    # Sort by keyword count (ascending)
    classes_with_counts.sort(key=lambda x: x[1])
    
    # Return classes with few keywords first
    priority_classes = [name for name, count in classes_with_counts if count < min_keywords]
    
    return priority_classes


if __name__ == "__main__":
    """Example usage."""
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    
    from src.data_preprocessing import DataLoader
    
    # Load data
    print("Loading data...")
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
    
    print(f"\nPriority classes (< 5 keywords): {len(priority_classes)}")
    
    # Expand keywords (will stop at max_calls)
    expanded_keywords = expander.expand_all_keywords(
        data_loader.class_keywords,
        priority_classes=priority_classes[:100]  # Expand top 100 priority classes
    )
    
    print("\nDone! Check logs/llm_calls/ for all prompts and responses.")