# src/evaluate_zero_shot.py

import sys
import os
from typing import List, Dict, Any
import logging
from tqdm import tqdm

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import DataLoader
from src.models.zero_shot import ZeroShotLanguageIdentifier

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    try:
        # Initialize data loader
        logger.info("Loading dataset...")
        data_loader = DataLoader()
        texts, labels = data_loader.load_dataset()
        
        # Initialize model
        logger.info("Initializing zero-shot model...")
        model = ZeroShotLanguageIdentifier()
        
        # Run evaluation
        logger.info("Starting evaluation...")
        report = model.evaluate(texts, labels)
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro avg']['f1-score']:.4f}")
        print("\nPer-language results:")
        for lang in sorted(set(labels)):
            if lang in report:
                print(f"{lang:>15}: F1 = {report[lang]['f1-score']:.4f}")
        
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    main()
