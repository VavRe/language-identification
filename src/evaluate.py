# evaluate_all.py
import os
import torch
import json
import numpy as np
from src.utils import DataLoader, ResultsHandler
from src.lstm import LSTMClassifier, LanguageDataset
from src.mlp import MLPClassifier  # Add this import
from src.transformer import TransformerClassifier, TransformerDataset
from transformers import XLMRobertaTokenizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import joblib
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm 


import warnings
warnings.filterwarnings('ignore', message='You are using `torch.load`.*')


def load_lstm_model(model_path):
    """Load LSTM model with its vocabulary and label encoder"""
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get the vocabulary and label encoder
        vocab = checkpoint['vocab']
        label_encoder = checkpoint['label_encoder']
        
        # Get model parameters from state dict
        state_dict = checkpoint['model_state_dict']
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
        return model, vocab, label_encoder
    except Exception as e:
        print(f"Error loading LSTM model: {e}")
        return None, None, None

def load_mlp_model(model_path):
    """Load MLP model with its vocabulary and label encoder"""
    try:
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Get the vocabulary and label encoder
        vocab = checkpoint['vocab']
        label_encoder = checkpoint['label_encoder']
        
        # Get model parameters from state dict
        state_dict = checkpoint['model_state_dict']
        
        # Get dimensions from the saved weights
        vocab_size = state_dict['embedding.weight'].shape[0]
        embedding_dim = state_dict['embedding.weight'].shape[1]
        
        # Find output dimension (number of classes)
        output_layer_key = [k for k in state_dict.keys() if k.endswith('.weight')][-1]
        output_dim = state_dict[output_layer_key].shape[0]
        
        # Create model with same architecture
        model = MLPClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=[512, 256, 128],  # Use same hidden dims as training
            output_dim=output_dim,
            dropout=0.3
        )
        model.load_state_dict(state_dict)
        return model, vocab, label_encoder
    except Exception as e:
        print(f"Error loading MLP model: {e}")
        return None, None, None


def load_transformer_model(model_path, model_name):
    """Load transformer model"""
    try:
        state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
        n_classes = state_dict['classifier.weight'].shape[0]
        
        model = TransformerClassifier(
            model_name=model_name,
            num_classes=n_classes,
            fine_tune_layers=3
        )
        model.load_state_dict(state_dict)
        return model
    except Exception as e:
        print(f"Error loading transformer model: {e}")
        return None

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
                accuracy = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average='macro')
                print(f"\n{model_name} Results:")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Macro F1: {f1:.4f}")
                results[model_name] = {'accuracy': accuracy, 'macro_f1': f1}
    
    # 2. Evaluate LSTM model
    print("\nEvaluating LSTM model...")
    lstm_path = "eval/deep_learning_results/models/lstm_language_detector_best.pt"
    
    if os.path.exists(lstm_path):
        lstm_model, vocab, label_encoder = load_lstm_model(lstm_path)
        if lstm_model is not None:
            lstm_model = lstm_model.to(device)
            lstm_model.eval()
            
            # Create test dataset using saved vocabulary and label encoder
            test_dataset = LanguageDataset(
                texts=X_test,
                labels=y_test,
                vocab=vocab  # Use saved vocabulary
            )
            
            test_loader = TorchDataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False
            )
            
            # Evaluation
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating LSTM"):
                    inputs = batch['input'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = lstm_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert predictions back to original labels
            y_pred = test_dataset.label_encoder.inverse_transform(all_preds)
            y_true = test_dataset.label_encoder.inverse_transform(all_labels)
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"\nLSTM Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {f1:.4f}")
            
            results['lstm'] = {'accuracy': accuracy, 'macro_f1': f1}
    else:
        print(f"LSTM model not found at {lstm_path}")
        
        # 3. Evaluate MLP model
    print("\nEvaluating MLP model...")
    mlp_path = "eval/deep_learning_results/models/mlp_language_detector_best.pt"
    
    if os.path.exists(mlp_path):
        mlp_model, vocab, label_encoder = load_mlp_model(mlp_path)
        if mlp_model is not None:
            mlp_model = mlp_model.to(device)
            mlp_model.eval()
            
            # Create test dataset using saved vocabulary and label encoder
            test_dataset = LanguageDataset(
                texts=X_test,
                labels=y_test,
                vocab=vocab
            )
            
            test_loader = TorchDataLoader(
                test_dataset,
                batch_size=64,
                shuffle=False
            )
            
            # Evaluation
            correct = 0
            total = 0
            all_preds = []
            all_labels = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating MLP"):
                    inputs = batch['input'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = mlp_model(inputs)
                    _, predicted = torch.max(outputs.data, 1)
                    
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    all_preds.extend(predicted.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            # Convert predictions back to original labels
            y_pred = test_dataset.label_encoder.inverse_transform(all_preds)
            y_true = test_dataset.label_encoder.inverse_transform(all_labels)
            
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            print(f"\nMLP Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {f1:.4f}")
            
            results['mlp'] = {'accuracy': accuracy, 'macro_f1': f1}
    else:
        print(f"MLP model not found at {mlp_path}")
    
    # 3. Evaluate Transformer model
    print("\nEvaluating Transformer model...")
    transformer_path = "eval/transformer_results/models/transformer_best.pt"
    model_name = '/home/v_rahimzadeh/hf_models/models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089'
    
    if os.path.exists(transformer_path):
        transformer_model = load_transformer_model(transformer_path, model_name)
        if transformer_model is not None:
            transformer_model = transformer_model.to(device)
            transformer_model.eval()
            
            tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(y_train)  # Fit on training data
            y_test_encoded = label_encoder.transform(y_test)
            
            test_dataset = TransformerDataset(X_test, y_test_encoded, tokenizer)
            test_loader = TorchDataLoader(test_dataset, batch_size=16)
            
            all_preds = []
            
            with torch.no_grad():
                for batch in tqdm(test_loader, desc="Evaluating Transformer"):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    outputs = transformer_model(input_ids, attention_mask)
                    _, predicted = torch.max(outputs.data, 1)
                    all_preds.extend(predicted.cpu().numpy())
            
            y_pred = label_encoder.inverse_transform(all_preds)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='macro')
            
            print(f"\nTransformer Results:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Macro F1: {f1:.4f}")
            
            results['transformer'] = {'accuracy': accuracy, 'macro_f1': f1}
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
