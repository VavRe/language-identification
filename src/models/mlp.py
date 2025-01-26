import torch
import torch.nn as nn
from typing import List, Dict, Any
from torch.utils.data import DataLoader
import os
from src.utils import EarlyStopping, get_path

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dims: List[int], output_dim: int, dropout: float):
        """
        MLP-based language classifier
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Number of classes
            dropout (float): Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.flatten = nn.Flatten()
        
        # Calculate input dimension for first linear layer
        self.input_dim = embedding_dim * 100  # 100 is sequence length
        
        # Build layers dynamically
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        flattened = self.flatten(embedded)
        return self.model(flattened)

def train_mlp(trainer, train_loader: DataLoader,
            val_loader: DataLoader,
            vocab_size: int,
            n_classes: int,
            config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Train MLP model
    """
    model = MLPClassifier(
        vocab_size=vocab_size,
        embedding_dim=config.get('embedding_dim', 300),
        hidden_dims=config.get('hidden_dims', [512, 256, 128]),
        output_dim=n_classes,
        dropout=config.get('dropout', 0.3)
    ).to(trainer.device)
    
    results = trainer._train_model(config.get('model_name', 'mlp'), 
                                 model, 
                                 train_loader, 
                                 val_loader, 
                                 config)
    
    # Save the model with proper state dict
    model_path = results['model_path']
    trainer.save_model(
        model,
        model_path,
        vocab=train_loader.dataset.vocab,
        label_encoder=train_loader.dataset.label_encoder
    )
    
    return results

def load_mlp_model(trainer, model_path: str, config: Dict[str, Any]) -> nn.Module:
    """
    Load a saved MLP model
    """
    try:
        checkpoint = torch.load(model_path)
        state_dict = checkpoint['model_state_dict']
        
        # Get vocab size from embedding layer
        vocab_size = state_dict['embedding.weight'].shape[0]
        embedding_dim = state_dict['embedding.weight'].shape[1]
        
        # Get output dimension from final layer
        # Find the last weight tensor in state dict that ends with '.weight'
        output_layer_key = [k for k in state_dict.keys() if k.endswith('.weight')][-1]
        output_dim = state_dict[output_layer_key].shape[0]
        
        # Get hidden dimensions by analyzing the model structure
        hidden_dims = []
        current_dim = embedding_dim * 100  # Initial flattened dimension
        layer_idx = 0
        
        while f'model.{layer_idx}.weight' in state_dict:
            next_dim = state_dict[f'model.{layer_idx}.weight'].shape[0]
            if layer_idx < len(state_dict) - 3:  # Not the last layer
                hidden_dims.append(next_dim)
            layer_idx += 3  # Skip ReLU and Dropout layers
        
        # Create model with inferred architecture
        model = MLPClassifier(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_dims=hidden_dims,
            output_dim=output_dim,
            dropout=config.get('dropout', 0.3)
        ).to(trainer.device)
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading MLP model: {str(e)}")
        if 'checkpoint' in locals():
            print(f"Checkpoint keys: {checkpoint.keys()}")
            print(f"State dict keys: {state_dict.keys() if 'state_dict' in locals() else 'No state dict'}")
        raise



import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

class MLPClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, 
                 hidden_dims: List[int], output_dim: int, dropout: float):
        """
        MLP-based language classifier
        Args:
            vocab_size (int): Size of vocabulary
            embedding_dim (int): Dimension of embeddings
            hidden_dims (List[int]): List of hidden layer dimensions
            output_dim (int): Number of classes
            dropout (float): Dropout rate
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.flatten = nn.Flatten()
        
        # Calculate input dimension for first linear layer
        self.input_dim = embedding_dim * 100  # 100 is sequence length
        
        # Build layers dynamically
        layers = []
        prev_dim = self.input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Add final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        embedded = self.embedding(text)
        flattened = self.flatten(embedded)
        return self.model(flattened)


class MLPTrainer:
    def __init__(self, model_dir: str = get_path("results","models","mlp")):
        """
        Trainer for MLP model
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
        self.writer = None  # Will be initialized in train

    def train(self,
              train_loader: DataLoader,
              val_loader: Optional[DataLoader],
              vocab_size: int,
              n_classes: int,
              config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train the MLP model
        Args:
            train_loader (DataLoader): Training data loader
            val_loader (DataLoader): Validation data loader
            vocab_size (int): Size of vocabulary
            n_classes (int): Number of classes
            config (Dict[str, Any]): Model configuration
        Returns:
            Dict[str, Any]: Training results
        """
        # Initialize model
        model = MLPClassifier(
            vocab_size=vocab_size,
            embedding_dim=config.get('embedding_dim', 300),
            hidden_dims=config.get('hidden_dims', [512, 256, 128]),
            output_dim=n_classes,
            dropout=config.get('dropout', 0.3)
        ).to(self.device)
        
        # Initialize tensorboard writer
        model_name = config.get('model_name', 'mlp')
        self.writer = SummaryWriter(os.path.join(self.tensorboard_dir, model_name))
        
        # Optimizer and loss function
        optimizer = torch.optim.Adam(model.parameters(), lr=config.get('lr', 0.001))
        criterion = nn.CrossEntropyLoss()
        n_epochs = config.get('n_epochs', 10)
        
        # Early stopping
        early_stopping = EarlyStopping(
            patience=config.get('patience', 10),
            min_delta=config.get('min_delta', 1e-4),
            mode='min'
        )
        
        # Training loop
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
                self.writer.add_scalar('batch/train_loss', loss.item(), global_step)
                
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
        
        # Return training results
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
        """
        Evaluate the model on a dataset
        Args:
            model (nn.Module): Model to evaluate
            data_loader (DataLoader): Data loader for evaluation
            criterion (nn.Module): Loss function
        Returns:
            Tuple[float, float]: Average loss and accuracy
        """
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
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
        
        return total_loss/len(data_loader), 100.*correct/total

    def predict(self, model: nn.Module, data_loader: DataLoader) -> np.ndarray:
        """
        Make predictions using the trained model
        Args:
            model (nn.Module): Trained model
            data_loader (DataLoader): Data loader for prediction
        Returns:
            np.ndarray: Array of predictions
        """
        model.eval()
        all_predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Making predictions"):
                inputs = batch['input'].to(self.device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                all_predictions.extend(predicted.cpu().numpy())
        
        return np.array(all_predictions)

    def save_model(self, model: nn.Module, model_path: str, vocab=None, label_encoder=None):
        """
        Save the model to disk
        Args:
            model (nn.Module): Model to save
            model_path (str): Path to save the model
            vocab: Vocabulary dictionary
            label_encoder: Label encoder object
        """
        checkpoint = {
            'model_state_dict': model.state_dict(),
            'vocab': vocab,
            'label_encoder': label_encoder,
            'architecture_params': {
                'vocab_size': vocab.size() if vocab else None,
                'embedding_dim': model.embedding.embedding_dim,
                'hidden_dims': [layer.out_features for layer in model.model if isinstance(layer, nn.Linear)][:-1],
                'output_dim': model.model[-1].out_features,
                'dropout': model.model[-2].p if isinstance(model.model[-2], nn.Dropout) else 0.0
            }
        }
        torch.save(checkpoint, model_path)

    def load_model(self, model_path: str) -> nn.Module:
        """
        Load a saved model
        Args:
            model_path (str): Path to the saved model
        Returns:
            nn.Module: Loaded model
        """
        checkpoint = torch.load(model_path)
        architecture_params = checkpoint['architecture_params']
        
        model = MLPClassifier(
            vocab_size=architecture_params['vocab_size'],
            embedding_dim=architecture_params['embedding_dim'],
            hidden_dims=architecture_params['hidden_dims'],
            output_dim=architecture_params['output_dim'],
            dropout=architecture_params['dropout']
        ).to(self.device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        return model