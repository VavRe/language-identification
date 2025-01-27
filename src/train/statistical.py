# src/train/statistical.py
import sys
import os
import time
import yaml
from typing import Dict, Any, List
from sklearn.metrics import classification_report
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import DataLoader, ResultsHandler, get_path
from src.models.statistical_models import NgramModel, CharacterFrequency, MarkovChain

def load_configs() -> Dict[str, Any]:
    config_path = get_path("configs.yaml")
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs.get("statistical", {})

def run_statistical_models():
    print("Starting statistical models training...")
    start_time = time.time()

    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Load dataset
    texts, labels = data_loader.load_dataset()
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)

    # Define models
    models = [
        NgramModel(n=3),
        CharacterFrequency(),
        MarkovChain()
    ]

    for model in models:
        print(f"\nTraining {model.model_name}...")
        
        # Train
        training_stats = model.train(X_train, y_train)
        
        # Save model
        model.save()
        
        # Predict
        predictions = model.predict(X_test)
        
        # Generate and save results with zero_division parameter
        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0  # Add this parameter
        )
        
        # Save results
        final_results = {
            'training_stats': training_stats,
            'classification_report': report,
            'execution_time': time.time() - start_time
        }
        
        # Add some basic statistics about predictions
        prediction_stats = {
            'total_samples': len(predictions),
            'unique_predictions': len(set(predictions)),
            'prediction_distribution': {label: predictions.count(label) 
                                     for label in set(predictions)}
        }
        final_results['prediction_stats'] = prediction_stats
        
        # Print some basic statistics
        print(f"\nPrediction Statistics for {model.model_name}:")
        print(f"Total samples: {prediction_stats['total_samples']}")
        print(f"Unique predictions: {prediction_stats['unique_predictions']}")
        print("\nPrediction distribution:")
        for label, count in prediction_stats['prediction_distribution'].items():
            print(f"{label}: {count}")
        
        results_handler.save_results(
            final_results,
            model.model_name,
            "statistical"
        )
        
        # Plot confusion matrix
        results_handler.plot_confusion_matrix(
            y_test,
            predictions,
            list(set(labels)),
            model.model_name,
            "statistical"
        )

    print(f"\nExecution completed in {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    run_statistical_models()
