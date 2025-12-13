#!/usr/bin/env python3
"""
AWS SageMaker용 Silver Labels 생성 스크립트
"""
import sys
import yaml
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.silver_labeling.generate_silver_labels import SilverLabelGenerator
from src.data_preprocessing import DataLoader as DataPreprocessor
from src.utils.logger import setup_logger

def main():
    # Load config
    config_path = project_root / "config" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger("silver_labels", config['output']['log_dir'])
    
    # Paths from config
    data_dir = project_root / config['data']['data_dir']
    output_dir = project_root / "data" / "intermediate"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data...")
    data_loader = DataPreprocessor(data_dir=str(data_dir))
    data_loader.load_all()
    
    logger.info("=" * 80)
    logger.info("Silver Label Generation for AWS SageMaker")
    logger.info("=" * 80)
    logger.info(f"Data directory: {data_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Get silver labeling config
    sl_config = config.get('silver_labeling', {})
    
    # Initialize generator
    generator = SilverLabelGenerator(
        class_keywords=data_loader.class_keywords,
        class_to_id=data_loader.class_to_id,
        id_to_class=data_loader.id_to_class,
        min_confidence=sl_config.get('min_confidence', 0.1),
        embedding_model=sl_config.get('embedding_model', 'sentence-transformers/all-mpnet-base-v2'),
        use_keyword_matching=sl_config.get('use_keyword_matching', True),
        use_semantic_similarity=sl_config.get('use_semantic_similarity', True),
        keyword_weight=sl_config.get('keyword_weight', 0.3),
        similarity_weight=sl_config.get('similarity_weight', 0.7),
        similarity_threshold=sl_config.get('similarity_threshold', 0.5),
        batch_size=sl_config.get('batch_size', 32)
    )
    
    # Generate silver labels
    logger.info("\n[Step 1] Generating silver labels for training data...")
    train_labels, train_confidences = generator.generate_labels(
        data_loader.train_corpus,
        output_file=str(output_dir / "train_silver_labels.pkl")
    )
    logger.info(f"✅ Saved train labels")
    
    logger.info("\n[Step 2] Generating silver labels for test data...")
    test_labels, test_confidences = generator.generate_labels(
        data_loader.test_corpus,
        output_file=str(output_dir / "test_silver_labels.pkl")
    )
    logger.info(f"✅ Saved test labels")
    
    logger.info("\nNext step: Run training script")
    logger.info("  python3 scripts/train_with_config.py")

if __name__ == "__main__":
    main()
