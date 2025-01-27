import sys
import os
import time
from typing import Dict, Any
import yaml

import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import DataLoader, ResultsHandler
from src.models.lieterature import CavnarTrenkleClassifier, DunningClassifier

def load_configs() -> Dict[str, Any]:
    """Load configurations from configs.yaml"""
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 
        "configs.yaml"
    )
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs.get("literature", {})

def run_literature_models():
    print("Starting literature models training...")
    start_time = time.time()

    # Load configurations
    config = load_configs()
    print("Loaded configurations:", config)

    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Load dataset
    print("Loading dataset...")
    texts, labels = data_loader.load_dataset()
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
    
    # Define models
    models = [
        CavnarTrenkleClassifier(
            max_profile_size=config.get('max_profile_size', 400)
        ),
        DunningClassifier(
            smoothing=config.get('smoothing', 0.5)
        )
    ]

    # Train and evaluate each model
    for model in models:
        print(f"\nTraining {model.model_name}...")
        
        # Train
        training_stats = model.train(X_train, y_train)
        print(f"Training stats: {training_stats}")
        
        # Save model
        model.save()
        
        # Predict
        predictions = model.predict(X_test)
        
        # Generate and save results
        results_handler.generate_report(
            y_test, 
            predictions,
            list(set(labels)),
            model.model_name,
            "literature"
        )
        
        # Plot confusion matrix
        results_handler.plot_confusion_matrix(
            y_test,
            predictions,
            list(set(labels)),
            model.model_name,
            "literature"
        )
        
        print(f"Completed {model.model_name}")

    execution_time = time.time() - start_time
    print(f"\nTotal execution time: {execution_time:.2f} seconds")
    return True

if __name__ == "__main__":
    success = run_literature_models()
    print("\nExecution completed successfully!" if success else "\nExecution failed!")
