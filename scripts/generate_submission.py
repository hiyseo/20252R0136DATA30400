"""
Generate Kaggle submission file from predictions.
Format: studentID_final.csv with predicted class IDs (space-separated) per line.
Note: predict.py now generates CSV automatically. This script is for creating final submission file.
"""

import argparse
import pickle
from pathlib import Path
import sys
import shutil

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.utils.logger import setup_logger

logger = setup_logger("Submission")


def generate_submission(predictions_path: str, output_path: str, student_id: str = "2020320135"):
    """
    Generate Kaggle submission file.
    
    Args:
        predictions_path: Path to predictions pickle file
        output_path: Path to save submission CSV
        student_id: Student ID for filename
    """
    # Load predictions
    logger.info(f"Loading predictions from {predictions_path}")
    with open(predictions_path, 'rb') as f:
        results = pickle.load(f)
    
    pids = results['pids']
    predictions = results['predictions']
    
    logger.info(f"Loaded predictions for {len(pids)} samples")
    
    # Generate submission file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Writing submission to {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for pid, pred_classes in zip(pids, predictions):
            # Convert class indices to space-separated string
            class_str = ' '.join(map(str, sorted(pred_classes)))
            f.write(f"{class_str}\n")
    
    logger.info(f"Submission file saved: {output_file}")
    logger.info(f"Total lines: {len(pids)}")
    
    # Print sample
    logger.info("\n=== Sample Predictions (first 5) ===")
    for i in range(min(5, len(predictions))):
        logger.info(f"Line {i+1}: {' '.join(map(str, sorted(predictions[i])))}")


def main():
    parser = argparse.ArgumentParser(description="Generate Kaggle submission file")
    parser.add_argument('--predictions', type=str, required=True,
                       help='Path to predictions pickle or CSV file')
    parser.add_argument('--output', type=str, default='results/submissions/2020320135_final.csv',
                       help='Path to output submission file')
    parser.add_argument('--student_id', type=str, default='2020320135',
                       help='Student ID for filename')
    
    args = parser.parse_args()
    
    # Check if input is already CSV
    if args.predictions.endswith('.csv'):
        logger.info(f"Input is already CSV format: {args.predictions}")
        logger.info(f"Copying to final submission: {args.output}")
        shutil.copy(args.predictions, args.output)
        logger.info(f"✓ Submission file created: {args.output}")
    else:
        # Generate from PKL
        generate_submission(args.predictions, args.output, args.student_id)
    
    logger.info("\n=== Submission Ready ===")
    logger.info(f"Upload {args.output} to Kaggle")


if __name__ == '__main__':
    main()
