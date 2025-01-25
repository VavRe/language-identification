# evaluate_all.py
import os
import torch
import json
import numpy as np
from src.utils import DataLoader, ResultsHandler
from src.deep_learning import LSTMClassifier, LanguageDataset
from src.transformer import TransformerClassifier, TransformerDataset
from transformers import XLMRobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from torch.utils.data import DataLoader as TorchDataLoader

def load_lstm_model(model_path):
    """Load LSTM model"""
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        
        embedding_weight = state_dict['embedding.weight']
        vocab_size, embedding_dim = embedding_weight.shape
        hidden_dim = state_dict['lstm.weight_ih_l0'].shape[0] // 4
        n_classes = state_dict['fc.weight'].shape[0]
        
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim=n_classes,
            n_layers=2,
            bidirectional=True,
            dropout=0.3
        )
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None

def load_transformer_model(model_path):
    """Load transformer model"""
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        n_classes = state_dict['classifier.weight'].shape[0]
        
        model = TransformerClassifier(
            model_name='xlm-roberta-base',
            num_classes=n_classes,
            fine_tune_layers=3
        )
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading transformer model: {e}")
        return None

def evaluate_model(y_true, y_pred, model_name, label_encoder=None):
    """Calculate and print evaluation metrics"""
    # If label_encoder is provided, transform predictions back to original labels
    if label_encoder is not None:
        y_pred = label_encoder.inverse_transform(y_pred)
    
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n{model_name} Results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    
    return accuracy, macro_f1

def main():
    # Initialize components
    data_loader = DataLoader()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("Loading dataset...")
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
    
    # Create label encoder
    label_encoder = LabelEncoder()
    label_encoder.fit(labels)  # Fit on all labels
    y_test_encoded = label_encoder.transform(y_test)
    
    results = {}
    
    # 1. Evaluate Classical ML models
    print("\nEvaluating Classical ML models...")
    ml_models_dir = "eval/classical_ml_results/models/"
    if os.path.exists(ml_models_dir):
        for model_file in os.listdir(ml_models_dir):
            if model_file.endswith('.joblib'):
                model_name = model_file.replace('.joblib', '')
                model = joblib.load(os.path.join(ml_models_dir, model_file))
                y_pred = model.predict(X_test)
                acc, f1 = evaluate_model(y_test, y_pred, model_name)
                results[model_name] = {'accuracy': acc, 'macro_f1': f1}
    
    # 2. Evaluate LSTM model
    print("\nEvaluating LSTM model...")
    lstm_path = "eval/deep_learning_results/models/lstm_language_detector_best.pt"
    
    if os.path.exists(lstm_path):
        lstm_model = load_lstm_model(lstm_path)
        if lstm_model is not None:
            lstm_model = lstm_model.to(device)
            lstm_model.eval()
            
            test_dataset = LanguageDataset(X_test, y_test_encoded)  # Use encoded labels
            test_loader = TorchDataLoader(test_dataset, batch_size=64)
            
            y_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch['input'].to(device)
                    outputs = lstm_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.extend(predicted.cpu().numpy())
            
            acc, f1 = evaluate_model(y_test, y_pred, "LSTM", label_encoder)
            results['lstm'] = {'accuracy': acc, 'macro_f1': f1}
    else:
        print(f"LSTM model not found at {lstm_path}")
    
    # 3. Evaluate Transformer model
    print("\nEvaluating Transformer model...")
    transformer_path = "eval/transformer_results/models/transformer_best.pt"
    
    if os.path.exists(transformer_path):
        transformer_model = load_transformer_model(transformer_path)
        if transformer_model is not None:
            transformer_model = transformer_model.to(device)
            transformer_model.eval()
            
            tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
            test_dataset = TransformerDataset(X_test, y_test_encoded, tokenizer)  # Use encoded labels
            test_loader = TorchDataLoader(test_dataset, batch_size=16)
            
            y_pred = []
            with torch.no_grad():
                for batch in test_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = transformer_model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.extend(predicted.cpu().numpy())
            
            acc, f1 = evaluate_model(y_test, y_pred, "Transformer", label_encoder)
            results['transformer'] = {'accuracy': acc, 'macro_f1': f1}
    else:
        print(f"Transformer model not found at {transformer_path}")
    
    # Print final comparison
    if results:
        print("\n=== Final Model Comparison ===")
        print("\nModel\t\tAccuracy\tMacro F1")
        print("-" * 40)
        for model_name, metrics in results.items():
            print(f"{model_name:<15} {metrics['accuracy']:.4f}\t{metrics['macro_f1']:.4f}")
    else:
        print("\nNo models were successfully evaluated.")

if __name__ == "__main__":
    main()
