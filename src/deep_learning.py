# src/deep_learning.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import os
import json
from collections import defaultdict
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

def plot_confusion_matrix(y_true, y_pred):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.tight_layout()
    plt.close()
    return figure

# Add this new class for early stopping
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        """
        Early stopping handler
        Args:
            patience (int): Number of epochs to wait before stopping
            min_delta (float): Minimum change in monitored value to qualify as an improvement
            mode (str): 'min' for loss, 'max' for metrics like accuracy
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = float('inf') if mode == 'min' else float('-inf')
        self.early_stop = False
        self.min_delta *= 1 if mode == 'min' else -1

    def __call__(self, current_value):
        if self.mode == 'min':
            improved = current_value < self.best_value - self.min_delta
        else:
            improved = current_value > self.best_value + self.min_delta

        if improved:
            self.best_value = current_value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


class LanguageDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], 
                 max_length: int = 100, vocab: Optional[Dict[str, int]] = None):
        """
        Custom Dataset for language detection
        Args:
            texts (List[str]): List of text samples
            labels (List[str]): List of language labels
            max_length (int): Maximum sequence length
            vocab (Dict[str, int], optional): Vocabulary dictionary
        """
        self.texts = texts
        self.max_length = max_length
        
        # Create or use vocabulary
        if vocab is None:
            self.vocab = self._create_vocabulary(texts)
        else:
            self.vocab = vocab
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)
        
        # Convert texts to sequences
        self.sequences = [self._text_to_sequence(text) for text in texts]

    def _create_vocabulary(self, texts: List[str]) -> Dict[str, int]:
        """Create character-level vocabulary"""
        chars = set(''.join(texts))
        return {char: idx + 1 for idx, char in enumerate(chars)}  # 0 is reserved for padding

    def _text_to_sequence(self, text: str) -> List[int]:
        """Convert text to sequence of indices"""
        sequence = [self.vocab.get(char, 0) for char in text[:self.max_length]]
        padding = [0] * (self.max_length - len(sequence))
        return sequence + padding

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input': torch.tensor(self.sequences[idx], dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dim: int, output_dim: int, n_layers: int,
                 bidirectional: bool, dropout: float):
        """
        LSTM-based language classifier
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dim (int): Hidden dimension
            output_dim (int): Number of classes
            n_layers (int): Number of LSTM layers
            bidirectional (bool): Whether to use bidirectional LSTM
            dropout (float): Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers,
                           bidirectional=bidirectional, batch_first=True,
                           dropout=dropout if n_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        
        if self.lstm.bidirectional:
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        else:
            hidden = hidden[-1]
            
        return self.fc(self.dropout(hidden))

class MLPClassifier(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, dropout: float):
        """
        MLP-based language classifier
        Args:
            input_dim (int): Input dimension
            hidden_dims (List[int]): List of hidden dimensions
            output_dim (int): Number of classes
            dropout (float): Dropout rate
        """
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

class DeepLearningTrainer:
    def __init__(self, model_dir: str = "eval/deep_learning_results/models"):
        """
        Trainer for deep learning models
        Args:
            model_dir (str): Directory to save models
        """
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        # Create tensorboard directory
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tensorboard_dir = os.path.join(model_dir, 'runs', current_time)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = None  # Will be initialized in _train_model

    def _train_model(self,
                    model_name: str,
                    model: nn.Module,
                    train_loader: DataLoader,
                    val_loader: Optional[DataLoader],
                    config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generic training loop
        """
        # Initialize tensorboard writer
        self.writer = SummaryWriter(os.path.join(self.tensorboard_dir, model_name))
        
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        criterion = nn.CrossEntropyLoss()
        n_epochs = config.get('n_epochs', 10)
        
        # Initialize early stopping
        early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 1e-4),
            mode='min'
        )
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        best_model_state = None
        
        for epoch in range(n_epochs):
            # Training
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{n_epochs}')
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Log batch-level metrics
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('batch/train_loss', 
                                     loss.item(), 
                                     global_step)
                
                pbar.set_postfix({
                    'loss': train_loss/(pbar.n+1),
                    'acc': 100.*train_correct/train_total
                })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss/len(train_loader)
            epoch_train_acc = 100.*train_correct/train_total
            train_losses.append(epoch_train_loss)
            
            # Log epoch-level training metrics
            self.writer.add_scalar('epoch/train_loss', epoch_train_loss, epoch)
            self.writer.add_scalar('epoch/train_accuracy', epoch_train_acc, epoch)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)
                
                # Log validation metrics
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                self.writer.add_scalar('epoch/val_accuracy', val_acc, epoch)
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, 
                             os.path.join(self.model_dir, f'{model_name}_best.pt'))
                
                # Early stopping check
                if early_stopping(val_loss):
                    print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
            
            # Log learning rate
            self.writer.add_scalar('epoch/learning_rate',
                                 optimizer.param_groups[0]['lr'],
                                 epoch)
        
        # Close tensorboard writer
        self.writer.close()
        
        # Load best model if available
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Save training curves
        self._plot_training_curves(train_losses, val_losses, model_name)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if val_loader else None,
            'final_train_loss': train_losses[-1],
            'best_val_loss': best_val_loss if val_loader else None,
            'model_path': os.path.join(self.model_dir, f'{model_name}_best.pt'),
            'early_stopped': early_stopping.early_stop,
            'epochs_trained': len(train_losses)
        }

    def _evaluate(self,
                 model: nn.Module,
                 data_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        """Evaluate model"""
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                inputs = batch['input'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Log confusion matrix if writer exists
        if self.writer is not None:
            figure = plot_confusion_matrix(all_labels, all_preds)
            self.writer.add_figure('confusion_matrix', figure)
        
        return total_loss/len(data_loader), 100.*correct/total


    def train_lstm(self, 
                train_loader: DataLoader,
                val_loader: Optional[DataLoader],
                vocab_size: int,
                n_classes: int,
                config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train LSTM model
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            vocab_size (int): Size of vocabulary
            n_classes (int): Number of classes
            config (Dict[str, Any]): Model configuration
        Returns:
            Dict[str, Any]: Training results
        """
        model = LSTMClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_dim=config.get('hidden_dim', 256),
            output_dim=n_classes,
            n_layers=config.get('n_layers', 2),
            bidirectional=config.get('bidirectional', True),
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        return self._train_model(config.get('model_name', 'lstm'), model, 
                                train_loader, val_loader, config)

    def train_mlp(self,
                train_loader: DataLoader,
                val_loader: Optional[DataLoader],
                input_dim: int,
                n_classes: int,
                config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train MLP model
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            input_dim (int): Input dimension
            n_classes (int): Number of classes
            config (Dict[str, Any]): Model configuration
        Returns:
            Dict[str, Any]: Training results
        """
        model = MLPClassifier(
            input_dim=input_dim,
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            output_dim=n_classes,
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        return self._train_model(config.get('model_name', 'mlp'), model, 
                            train_loader, val_loader, config)

    def load_model(self, model_type: str, model_path: str, config: Dict[str, Any]) -> nn.Module:
        """
        Load a saved model
        Args:
            model_type (str): Type of model ('lstm' or 'mlp')
            model_path (str): Path to saved model
            config (Dict[str, Any]): Model configuration
        Returns:
            nn.Module: Loaded model
        """
        if model_type == 'lstm':
            model = LSTMClassifier(
                vocab_size=config.get('vocab_size'),
                embedding_dim=config.get('embedding_dim', 300),
                hidden_dim=config.get('hidden_dim', 256),
                output_dim=config.get('n_classes'),
                n_layers=config.get('n_layers', 2),
                bidirectional=config.get('bidirectional', True),
                dropout=config.get('dropout', 0.3)
            ).to(self.device)
        elif model_type == 'mlp':
            model = MLPClassifier(
                input_dim=config.get('input_dim'),
                hidden_dims=config.get('hidden_dims', [512, 256, 128]),
                output_dim=config.get('n_classes'),
                dropout=config.get('dropout', 0.3)
            ).to(self.device)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.load_state_dict(torch.load(model_path))
        model.eval()
        return model

    def save_model(self, model: nn.Module, model_path: str):
        """
        Save model to disk
        Args:
            model (nn.Module): Model to save
            model_path (str): Path to save model
        """
        torch.save(model.state_dict(), model_path)