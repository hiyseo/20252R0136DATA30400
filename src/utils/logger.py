"""
Logging utilities for training and evaluation.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path


def setup_logger(name: str = "DATA304", log_dir: str = "logs", level=logging.INFO):
    """
    Setup logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create log directory
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        log_path / f"{name}_{timestamp}.log"
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger