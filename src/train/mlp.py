import sys
import os
import yaml  # Import the yaml library

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import DataLoader, ResultsHandler, LanguageDataset
from src.models.mlp import MLPTrainer, MLPClassifier
from torch.utils.data import DataLoader as TorchDataLoader
import torch
from sklearn.metrics import classification_report
import numpy as np
import time
from typing import Dict, Any


def load_configs() -> Dict[str, Any]:
    """
    Load configurations from configs.yaml
    Returns:
        Dict[str, Any]: Configurations for the MLP model
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs.yaml")
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs.get("mlp", {})  # Extract the 'mlp' section


def run_mlp():
    print("Starting MLP model training...")
    start_time = time.time()

    # Load configurations
    mlp_config = load_configs()
    print("Loaded MLP configurations:", mlp_config)

    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Load and prepare data
    print("Loading dataset...")
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
    
    # Create datasets
    print("Preparing datasets...")
    train_dataset = LanguageDataset(X_train, y_train)
    test_dataset = LanguageDataset(
        X_test, 
        y_test, 
        vocab=train_dataset.vocab, 
        label_encoder=train_dataset.label_encoder  # Share label encoder
    )
    
    # Create data loaders
    num_workers = min(os.cpu_count(), 4)  # Use up to 4 workers
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=mlp_config.get("batch_size", 64), 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Speed up data transfer to GPU
    )
    test_loader = TorchDataLoader(
        test_dataset, 
        batch_size=mlp_config.get("batch_size", 64),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize trainer
    mlp_trainer = MLPTrainer(model_dir="results/models/mlp")

    # MLP Configuration and Training
    print("\nTraining MLP model...")
    mlp_results = mlp_trainer.train_mlp(
        train_loader=train_loader,
        val_loader=test_loader,
        vocab_size=len(train_dataset.vocab) + 1,
        n_classes=len(train_dataset.label_encoder.classes_),
        config=mlp_config
    )
    
    # Load and evaluate best MLP model
    best_mlp_path = mlp_results['model_path']
    if os.path.exists(best_mlp_path):
        print("\nEvaluating best MLP model...")
        best_mlp = mlp_trainer.load_model('mlp', best_mlp_path, mlp_config)
        
        # Save model with vocabulary and label encoder
        mlp_trainer.save_model(
            best_mlp,
            best_mlp_path,
            vocab=train_dataset.vocab,
            label_encoder=train_dataset.label_encoder
        )
        
        # Make predictions
        mlp_predictions = mlp_trainer.predict(best_mlp, test_loader)
        mlp_predictions_labels = [train_dataset.label_encoder.classes_[i] for i in mlp_predictions]
        
        # Generate classification report
        mlp_report = classification_report(
            y_test,
            mlp_predictions_labels,
            target_names=train_dataset.label_encoder.classes_,
            output_dict=True
        )
        mlp_results['classification_report'] = mlp_report

    # Save final results
    try:
        final_results = {
            'mlp': {
                **mlp_results,
                'config': mlp_config,
                'execution_time': time.time() - start_time,
                'num_parameters': sum(p.numel() for p in best_mlp.parameters()) if 'best_mlp' in locals() else None
            },
            'dataset_info': {
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'num_classes': len(train_dataset.label_encoder.classes_),
                'vocab_size': len(train_dataset.vocab) + 1
            }
        }
    except Exception as e:
        print(f"Error creating final results: {str(e)}")
        raise

    # Save results
    print("\nSaving results...")
    results_handler.save_results(final_results, "mlp_results", "mlp")
    
    print(f"\nExecution completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved in: {results_handler.base_path}/mlp_results/")
    print("\nTo view training progress, run:")
    print(f"tensorboard --logdir={mlp_trainer.tensorboard_dir}")
    
    return True


if __name__ == "__main__":
    success = run_mlp()
    print("\nTraining completed successfully!" if success else "\nTraining failed!")