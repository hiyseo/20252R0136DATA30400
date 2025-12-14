"""
AWS SageMaker training script with config support.
Uses train_baseline.py as the underlying training engine.
"""

import sys
import yaml
import argparse
from pathlib import Path

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from src.training.train_baseline import train_baseline_model
from src.utils.logger import setup_logger


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def config_to_args(config: dict) -> argparse.Namespace:
    """Convert config dict to argparse.Namespace for train_baseline compatibility."""
    args = argparse.Namespace()
    
    # Data
    args.train_labels_path = config['data']['train_labels_path']
    args.test_labels_path = config['data']['test_labels_path']
    args.max_length = config['data']['max_length']
    args.num_workers = config['data']['num_workers']
    
    # Model
    args.model_name = config['model']['model_name']
    args.dropout = config['model']['dropout']
    args.freeze_encoder = config['model']['freeze_encoder']
    
    # Training
    args.batch_size = config['training']['batch_size']
    args.num_epochs = config['training']['num_epochs']
    args.learning_rate = config['training']['learning_rate']
    args.weight_decay = config['training']['weight_decay']
    args.warmup_ratio = config['training']['warmup_ratio']
    args.loss_type = config['training']['loss_type']
    args.label_smoothing = config['training'].get('label_smoothing', 0.0)
    args.model_type = config['model']['model_type']
    
    # Self-training
    st_config = config.get('self_training', {})
    args.use_self_training = st_config.get('enabled', False)
    args.self_training_confidence = st_config.get('confidence_threshold', 0.7)
    args.self_training_iterations = st_config.get('max_iterations', 3)
    
    # Misc
    args.seed = config['misc']['seed']
    
    # Output dir: replace {model_type} placeholder with actual value
    output_dir = config['output']['output_dir']
    args.output_dir = output_dir.replace('{model_type}', args.model_type)
    
    # Training dir: replace {model_type} placeholder with actual value
    training_dir = config['output']['training_dir']
    args.training_dir = training_dir.replace('{model_type}', args.model_type)
    
    args.save_every = config['evaluation']['save_every']
    
    return args


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file')
    parser.add_argument('--model_name', type=str, help='Override model name')
    parser.add_argument('--batch_size', type=int, help='Override batch size')
    parser.add_argument('--num_epochs', type=int, help='Override num epochs')
    parser.add_argument('--learning_rate', type=float, help='Override learning rate')
    parser.add_argument('--use_self_training', action='store_true', 
                       help='Override self-training setting')
    
    cmd_args = parser.parse_args()
    
    # Load config
    logger = setup_logger("ConfigTraining")
    logger.info(f"Loading config from {cmd_args.config}")
    config = load_config(cmd_args.config)
    
    # Convert to args
    args = config_to_args(config)
    
    # Add student_id from config
    args.student_id = config.get('submission', {}).get('student_id', '2020320135')
    
    # Override with command line arguments
    if cmd_args.model_name:
        args.model_name = cmd_args.model_name
        logger.info(f"Overriding model_name: {cmd_args.model_name}")
    if cmd_args.batch_size:
        args.batch_size = cmd_args.batch_size
        logger.info(f"Overriding batch_size: {cmd_args.batch_size}")
    if cmd_args.num_epochs:
        args.num_epochs = cmd_args.num_epochs
        logger.info(f"Overriding num_epochs: {cmd_args.num_epochs}")
    if cmd_args.learning_rate:
        args.learning_rate = cmd_args.learning_rate
        logger.info(f"Overriding learning_rate: {cmd_args.learning_rate}")
    if cmd_args.use_self_training:
        args.use_self_training = True
        logger.info("Overriding use_self_training: True")
    
    logger.info("="*60)
    logger.info("Starting Training with Config")
    logger.info("="*60)
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Epochs: {args.num_epochs}")
    logger.info(f"Learning rate: {args.learning_rate}")
    logger.info(f"Loss type: {args.loss_type}")
    logger.info(f"Self-training: {args.use_self_training}")
    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Training dir: {args.training_dir}")
    logger.info("="*60)
    
    # Call train_baseline_model with args
    model = train_baseline_model(args)
    
    logger.info("\n" + "="*60)
    logger.info("Training Complete!")
    logger.info("="*60)
    logger.info(f"Model saved to: {args.output_dir}")
    
    return model


if __name__ == "__main__":
    main()
