import sys
import os
import yaml  # Import the yaml library

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.utils import DataLoader, ResultsHandler, get_path
from src.models.transformer import TransformerTrainer, TransformerDataset, TransformerClassifier
from torch.utils.data import DataLoader as TorchDataLoader
import torch
from sklearn.metrics import classification_report
import time
from transformers import XLMRobertaTokenizer
from typing import Dict, Any

def load_configs() -> Dict[str, Any]:
    """
    Load configurations from configs.yaml
    Returns:
        Dict[str, Any]: Configurations for the transformer model
    """
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "configs.yaml")
    with open(config_path, 'r') as file:
        configs = yaml.safe_load(file)
    return configs.get("transformer", {})  # Extract the 'transformer' section

def run_transformer():
    print("Starting transformer model testing...")
    start_time = time.time()

    # Load configurations
    transformer_config = load_configs()
    print("Loaded transformer configurations:", transformer_config)

    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    
    # Ensure texts are strings
    texts = [str(text) for text in texts]  # Convert all texts to strings
    
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    model_name = transformer_config.get("model_name", "xlm-roberta-base")
    
    if transformer_config.get("local_model", False):
        model_name = '/home/v_rahimzadeh/hf_models/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'

    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = TransformerDataset(
        texts=[str(text) for text in X_train],  # Ensure strings
        labels=y_train,
        tokenizer=tokenizer
    )
    
    test_dataset = TransformerDataset(
        texts=[str(text) for text in X_test],  # Ensure strings
        labels=y_test,
        tokenizer=tokenizer
    )
    
    # Create data loaders
    num_workers = min(os.cpu_count(), 4) if os.cpu_count() else 0
    train_loader = TorchDataLoader(
        train_dataset,
        batch_size=transformer_config.get("train_batch_size", 256),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    test_loader = TorchDataLoader(
        test_dataset,
        batch_size=transformer_config.get("test_batch_size", 16),
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    # Initialize trainer
    trainer_path = get_path('results', 'models', 'transformer')
    trainer = TransformerTrainer(trainer_path)
        
    # Configure training
    config = {
        'model_name': model_name,  # Ensure string
        'fine_tune_layers': int(transformer_config.get("fine_tune_layers", 3)),  # Ensure integer
        'lr': float(transformer_config.get("lr", 2e-5)),  # Ensure float
        'weight_decay': float(transformer_config.get("weight_decay", 0.01)),  # Ensure float
        'n_epochs': int(transformer_config.get("n_epochs", 1)),  # Ensure integer
        'patience': int(transformer_config.get("patience", 3)),  # Ensure integer
        'min_delta': float(transformer_config.get("min_delta", 1e-4)),  # Ensure float
        'warmup_ratio': float(transformer_config.get("warmup_ratio", 0.1)),  # Ensure float
        'batch_size': int(transformer_config.get("batch_size", 16))  # Ensure integer
    }
    
    # Train model
    print("\nTraining transformer model...")
    results = trainer.train(
        train_loader=train_loader,
        val_loader=test_loader,
        n_classes=len(train_dataset.label_encoder.classes_),
        config=config
    )
    
    # Load best model and evaluate
    print("\nEvaluating best model...")
    best_model = TransformerClassifier(
        model_name=model_name,
        num_classes=len(train_dataset.label_encoder.classes_),
        fine_tune_layers=config['fine_tune_layers']
    ).to(trainer.device)
    
    best_model.load_state_dict(torch.load(results['model_path']))
    predictions = trainer.predict(best_model, test_loader)
    
    # Convert y_test to numeric format using the label encoder
    y_test_numeric = train_dataset.label_encoder.transform(y_test)
    
    # Generate classification report
    report = classification_report(
        y_test_numeric,  # Use numeric format for y_true
        predictions,     # Predictions are already in numeric format
        target_names=train_dataset.label_encoder.classes_,  # Use string labels for report
        output_dict=True
    )
    
    # Prepare final results
    final_results = {
        'training_results': results,
        'classification_report': report,
        'config': config,
        'execution_time': time.time() - start_time,
        'model_size': sum(p.numel() for p in best_model.parameters())
    }
    
    # Save results
    print("\nSaving results...")
    results_handler.save_results(final_results, "transformer_results", "transformer")
    
    print(f"\nExecution completed in {time.time() - start_time:.2f} seconds")
    print(f"Results saved in: {results_handler.base_path}/transformer_results/")
    print("\nTo view training progress, run:")
    print(f"tensorboard --logdir={trainer.tensorboard_dir}")
    
    return True

if __name__ == "__main__":
    success = run_transformer()
    print("\nTest completed successfully!" if success else "\nTest failed!")