import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from src.utils import DataLoader, ResultsHandler, LanguageDataset, EarlyStopping
from src.models.lstm import LSTMTrainer
from src.models.mlp import MLPTrainer
from src.models.mlp import train_mlp, load_mlp_model  # Add this import
from torch.utils.data import DataLoader as TorchDataLoader
import torch
from sklearn.metrics import classification_report
import numpy as np
import time
import os
import torch

def test_deep_learning():
    print("Starting deep learning models testing...")
    start_time = time.time()

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
    # Create data loaders with num_workers for faster data loading
    num_workers = min(os.cpu_count(), 4)  # Use up to 4 workers
    train_loader = TorchDataLoader(
        train_dataset, 
        batch_size=64, 
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()  # Speed up data transfer to GPU
    )
    test_loader = TorchDataLoader(
        test_dataset, 
        batch_size=64,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize trainer
       # Initialize early stopping
  
    lstm_trainer = LSTMTrainer()
    
    # LSTM Configuration and Training
    lstm_config = {
        'embedding_dim': 300,
        'hidden_dim': 256,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'lr': 0.001,
        'n_epochs': 1,
        'patience': 10,
        'min_delta': 1e-4,
        'batch_size': 64,
        'model_name': 'lstm_language_detector'
    }
    
    print("\nTraining LSTM model...")
    lstm_results = lstm_trainer.train_lstm(
        train_loader=train_loader,
        val_loader=test_loader,
        vocab_size=len(train_dataset.vocab) + 1,
        n_classes=len(train_dataset.label_encoder.classes_),
        config=lstm_config
    )
    
    
    # Load and evaluate best LSTM model
    best_lstm_path = lstm_results['model_path']
    if os.path.exists(best_lstm_path):
        print("\nEvaluating best LSTM model...")
        best_lstm = lstm_trainer.load_model('lstm', best_lstm_path, lstm_config)
        lstm_trainer.save_model(
            best_lstm,
            best_lstm_path,
            vocab=train_dataset.vocab,
            label_encoder=train_dataset.label_encoder
        )
        lstm_predictions = lstm_trainer.predict(best_lstm, test_loader)
        lstm_predictions_labels = [train_dataset.label_encoder.classes_[i] for i in lstm_predictions]
        lstm_report = classification_report(
            y_test,
            lstm_predictions_labels,
            target_names=train_dataset.label_encoder.classes_,
            output_dict=True
        )
        lstm_results['classification_report'] = lstm_report



# Initialize the MLPTrainer
    mlp_trainer = MLPTrainer(model_dir="results/models/mlp")

   
    mlp_config = {
        'embedding_dim': 300,
        'hidden_dims': [512, 256, 128],
        'dropout': 0.3,
        'lr': 0.001,
        'n_epochs': 20,
        'patience': 10,
        'min_delta': 1e-4,
        'batch_size': 64,
        'model_name': 'mlp_language_detector'
    }

    
    print("\nTraining MLP model...")
    mlp_results = mlp_trainer.train(
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
        try:
            best_mlp = load_mlp_model(mlp_trainer, best_mlp_path, mlp_config)
            
            # Save model with vocabulary and label encoder
            mlp_trainer.save_model(
                best_mlp,
                best_mlp_path,
                vocab=train_dataset.vocab,
                label_encoder=train_dataset.label_encoder
            )
            
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
        except Exception as e:
            print(f"Error in MLP evaluation: {str(e)}")
            mlp_results['error'] = str(e)

    # Save final results for both models
    try:
        final_results = {
            'lstm': {
                **lstm_results,
                'config': lstm_config,
                'execution_time': time.time() - start_time,
                'num_parameters': sum(p.numel() for p in best_lstm.parameters()) if 'best_lstm' in locals() else None
            },
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
    results_handler.save_results(final_results, "model_comparison", "deep_learning")
    
    # print(f"\nExecution completed in {execution_time:.2f} seconds")
    print(f"Results saved in: {results_handler.base_path}/deep_learning_results/")
    print("\nTo view training progress, run:")
    # print(f"tensorboard --logdir={trainer.tensorboard_dir}")
    
    return True

if __name__ == "__main__":
        success = test_deep_learning()
        print("\nTest completed successfully!" if success else "\nTest failed!")
