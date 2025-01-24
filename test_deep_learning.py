# test_deep_learning.py
from src.utils import DataLoader, ResultsHandler
from src.deep_learning import LanguageDataset, DeepLearningTrainer
from torch.utils.data import DataLoader as TorchDataLoader
import torch
from sklearn.metrics import classification_report
import numpy as np
import time
import os

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
    test_dataset = LanguageDataset(X_test, y_test, vocab=train_dataset.vocab)
    
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
    trainer = DeepLearningTrainer()
    
    # LSTM Configuration
    lstm_config = {
        'embedding_dim': 300,
        'hidden_dim': 256,
        'n_layers': 2,
        'bidirectional': True,
        'dropout': 0.3,
        'lr': 0.001,
        'n_epochs': 100,
        'patience': 10,
        'min_delta': 1e-4,
        'batch_size': 64,
        'model_name': 'lstm_language_detector'
    }
    
    print("\nTraining LSTM model...")
    lstm_results = trainer.train_lstm(
        train_loader=train_loader,
        val_loader=test_loader,
        vocab_size=len(train_dataset.vocab) + 1,
        n_classes=len(train_dataset.label_encoder.classes_),
        config=lstm_config
    )
    
    # Load best LSTM model and evaluate
    best_lstm_path = lstm_results['model_path']
    if os.path.exists(best_lstm_path):
        print("\nEvaluating best LSTM model...")
        best_lstm = trainer.load_model('lstm', best_lstm_path, lstm_config)
        lstm_predictions = trainer.predict(best_lstm, test_loader)
        
        # Generate classification report
        lstm_report = classification_report(
            y_test,
            lstm_predictions,
            target_names=train_dataset.label_encoder.classes_,
            output_dict=True
        )
        lstm_results['classification_report'] = lstm_report
    
    # MLP Configuration
    mlp_config = {
        'hidden_dims': [512, 256, 128],
        'dropout': 0.3,
        'lr': 0.001,
        'n_epochs': 100,
        'patience': 10,
        'min_delta': 1e-4,
        'batch_size': 64,
        'model_name': 'mlp_language_detector'
    }
    
    print("\nTraining MLP model...")
    mlp_results = trainer.train_mlp(
        train_loader=train_loader,
        val_loader=test_loader,
        input_dim=lstm_config['embedding_dim'],
        n_classes=len(train_dataset.label_encoder.classes_),
        config=mlp_config
    )
    
    # Load best MLP model and evaluate
    best_mlp_path = mlp_results['model_path']
    if os.path.exists(best_mlp_path):
        print("\nEvaluating best MLP model...")
        best_mlp = trainer.load_model('mlp', best_mlp_path, mlp_config)
        mlp_predictions = trainer.predict(best_mlp, test_loader)
        
        # Generate classification report
        mlp_report = classification_report(
            y_test,
            mlp_predictions,
            target_names=train_dataset.label_encoder.classes_,
            output_dict=True
        )
        mlp_results['classification_report'] = mlp_report
    
    # Add execution time
    execution_time = time.time() - start_time
    
    # Prepare final results
    final_results = {
        'lstm': {
            **lstm_results,
            'config': lstm_config,
            'execution_time': execution_time,
            'num_parameters': sum(p.numel() for p in best_lstm.parameters())
        },
        'mlp': {
            **mlp_results,
            'config': mlp_config,
            'execution_time': execution_time,
            'num_parameters': sum(p.numel() for p in best_mlp.parameters())
        },
        'dataset_info': {
            'train_samples': len(X_train),
            'test_samples': len(X_test),
            'num_classes': len(train_dataset.label_encoder.classes_),
            'vocab_size': len(train_dataset.vocab) + 1
        }
    }
    
    # Save results
    print("\nSaving results...")
    results_handler.save_results(final_results, "model_comparison", "deep_learning")
    
    print(f"\nExecution completed in {execution_time:.2f} seconds")
    print(f"Results saved in: {results_handler.base_path}/deep_learning_results/")
    print("\nTo view training progress, run:")
    print(f"tensorboard --logdir={trainer.tensorboard_dir}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_deep_learning()
        print("\nTest completed successfully!" if success else "\nTest failed!")
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
