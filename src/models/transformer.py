# src/transformer_model.py
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaModel, 
    XLMRobertaTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import confusion_matrix

class EarlyStopping:
    def __init__(self, patience=3, min_delta=1e-4, mode='min'):
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

class TransformerClassifier(nn.Module):
    def __init__(self, model_name: str, num_classes: int, fine_tune_layers: int = 3):
        super().__init__()
        self.transformer = XLMRobertaModel.from_pretrained(model_name)
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        
        # Freeze all layers except the last fine_tune_layers
        total_layers = len(list(self.transformer.encoder.layer))
        for param in self.transformer.parameters():
            param.requires_grad = False
            
        if fine_tune_layers > 0:
            for layer in self.transformer.encoder.layer[total_layers-fine_tune_layers:]:
                for param in layer.parameters():
                    param.requires_grad = True
            
            # Always fine-tune pooler and embeddings
            for param in self.transformer.pooler.parameters():
                param.requires_grad = True
            for param in self.transformer.embeddings.parameters():
                param.requires_grad = True

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_classes)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        return self.classifier(pooled_output)

class TransformerDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[str], tokenizer, max_length: int = 128):
        """
        Dataset for transformer models
        Args:
            texts (List[str]): List of text samples
            labels (List[str]): List of language labels
            tokenizer: Transformer tokenizer
            max_length (int): Maximum sequence length
        """
        # Ensure all texts are strings
        self.texts = [str(text) for text in texts]
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(labels)
        
        print("Tokenizing texts...")
        try:
            self.encodings = self.tokenizer(
                self.texts,
                max_length=max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
        except Exception as e:
            print(f"Error during tokenization: {str(e)}")
            print(f"First few texts: {self.texts[:5]}")
            raise

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            'input_ids': self.encodings['input_ids'][idx],
            'attention_mask': self.encodings['attention_mask'][idx],
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

class TransformerTrainer:
    def __init__(self, model_dir: str = "results/models/transformer"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        
        current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
        self.tensorboard_dir = os.path.join(model_dir, 'runs', current_time)
        os.makedirs(self.tensorboard_dir, exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.writer = None

    def train(self,
             train_loader: DataLoader,
             val_loader: Optional[DataLoader],
             n_classes: int,
             config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Train transformer model
        """
        print(f"Using device: {self.device}")
        self.writer = SummaryWriter(os.path.join(self.tensorboard_dir, config.get('model_name', 'transformer')))
        
        model = TransformerClassifier(
            model_name=config.get('model_name', 'xlm-roberta-base'),
            num_classes=n_classes,
            fine_tune_layers=config.get('fine_tune_layers', 3)
        ).to(self.device)
        
        # Optimizer with different learning rates
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in model.named_parameters() if 'classifier' not in n],
                'lr': config.get('lr', 2e-5)
            },
            {
                'params': [p for n, p in model.named_parameters() if 'classifier' in n],
                'lr': config.get('lr', 2e-5) * 10
            }
        ]
        
        optimizer = AdamW(
            optimizer_grouped_parameters,
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # Learning rate scheduler
        num_training_steps = len(train_loader) * config.get('n_epochs', 10)
        num_warmup_steps = int(num_training_steps * config.get('warmup_ratio', 0.1))
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
        
        criterion = nn.CrossEntropyLoss()
        early_stopping = EarlyStopping(
            patience=config.get('patience', 3),
            min_delta=config.get('min_delta', 1e-4)
        )
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        best_model_state = None
        
        for epoch in range(config.get('n_epochs', 10)):
            model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["n_epochs"]}')
            for batch_idx, batch in enumerate(pbar):
                optimizer.zero_grad()
                
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                loss.backward()
                optimizer.step()
                scheduler.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
                
                # Log batch metrics
                global_step = epoch * len(train_loader) + batch_idx
                self.writer.add_scalar('batch/train_loss', loss.item(), global_step)
                
                pbar.set_postfix({
                    'loss': train_loss/(batch_idx+1),
                    'acc': 100.*train_correct/train_total
                })
            
            # Calculate epoch metrics
            epoch_train_loss = train_loss/len(train_loader)
            epoch_train_acc = 100.*train_correct/train_total
            train_losses.append(epoch_train_loss)
            
            # Log epoch metrics
            self.writer.add_scalar('epoch/train_loss', epoch_train_loss, epoch)
            self.writer.add_scalar('epoch/train_accuracy', epoch_train_acc, epoch)
            self.writer.add_scalar('epoch/learning_rate', scheduler.get_last_lr()[0], epoch)
            
            # Validation
            if val_loader is not None:
                val_loss, val_acc = self._evaluate(model, val_loader, criterion)
                val_losses.append(val_loss)
                
                self.writer.add_scalar('epoch/val_loss', val_loss, epoch)
                self.writer.add_scalar('epoch/val_accuracy', val_acc, epoch)
                
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict()
                    torch.save(best_model_state, 
                             os.path.join(self.model_dir, 'transformer_best.pt'))
                
                if early_stopping(val_loss):
                    print(f'\nEarly stopping triggered after epoch {epoch+1}')
                    break
        
        self.writer.close()
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if val_loader else None,
            'final_train_loss': train_losses[-1],
            'best_val_loss': best_val_loss if val_loader else None,
            'model_path': os.path.join(self.model_dir, 'transformer_best.pt'),
            'early_stopped': early_stopping.early_stop,
            'epochs_trained': len(train_losses)
        }

    def _evaluate(self,
                 model: nn.Module,
                 data_loader: DataLoader,
                 criterion: nn.Module) -> Tuple[float, float]:
        model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        if self.writer is not None:
            cm = confusion_matrix(all_labels, all_preds)
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
            plt.title('Confusion Matrix')
            plt.colorbar()
            self.writer.add_figure('confusion_matrix', fig)
            plt.close()
        
        return total_loss/len(data_loader), 100.*correct/total

    def predict(self, model: nn.Module, data_loader: DataLoader) -> np.ndarray:
        model.eval()
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                outputs = model(input_ids, attention_mask)
                _, predicted = torch.max(outputs.data, 1)
                predictions.extend(predicted.cpu().numpy())
        
        return np.array(predictions)
