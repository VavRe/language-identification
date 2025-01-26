
import os, torch
import pandas as pd
import kagglehub
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from tqdm import tqdm
from torch.serialization import add_safe_globals
from sklearn.preprocessing._label import LabelEncoder
from torch.utils.data import Dataset, DataLoader


add_safe_globals([LabelEncoder])


import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def get_path(*path_segments):
    """
    Construct an absolute path relative to the project root.
    Args:
        *path_segments: Path segments to join (e.g., "results", "models").
    Returns:
        str: Absolute path.
    """
    return os.path.join(PROJECT_ROOT, *path_segments)

class DataLoader:
    @staticmethod
    def download_dataset() -> str:
        """
        Check for dataset in ../dataset/dataset.csv, if not exists, download from Kaggle
        Returns:
            str: Path to the dataset
        """
        # Check local path first
        local_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                 "dataset", "dataset.csv")
        
        # If local file exists, return its path
        if os.path.exists(local_path):
            print(f"Dataset found locally at: {local_path}")
            return local_path
            
        # If not, create directory and download
        try:
            # Create dataset directory if it doesn't exist
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # Download from Kaggle
            print("Downloading dataset from Kaggle...")
            kaggle_path = kagglehub.dataset_download("zarajamshaid/language-identification-datasst")
            
            # Copy file to local directory
            import shutil
            shutil.copy2(kaggle_path + "/dataset.csv", local_path)
            
            print(f"Dataset downloaded and saved to: {local_path}")
            return local_path
            
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return None

    @staticmethod
    def load_dataset(csv_path: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load dataset from CSV file
        Args:
            csv_path (str, optional): Path to CSV file. 
                If None, looks in ../dataset/dataset.csv
        Returns:
            Tuple[np.ndarray, np.ndarray]: Texts and labels
        """
        if csv_path is None:
            csv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                                  "dataset", "dataset.csv")
        
        try:
            df = pd.read_csv(csv_path)
            # Basic preprocessing
            df['Text'] = df['Text'].str.strip()
            df['Language'] = df['language'].str.strip()
            
            # Remove empty texts
            df = df[df['Text'].str.len() > 0]
            
            # Convert to numpy arrays
            texts = df['Text'].values
            labels = df['Language'].values
            
            print(f"Loaded {len(texts)} samples with {len(np.unique(labels))} unique languages")
            return texts, labels
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None, None

    @staticmethod
    def prepare_data(texts: np.ndarray, 
                    labels: np.ndarray, 
                    test_size: float = 0.2, 
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train and test sets
        Args:
            texts (np.ndarray): Array of texts
            labels (np.ndarray): Array of labels
            test_size (float, optional): Test set size. Defaults to 0.2.
            random_state (int, optional): Random state. Defaults to 42.
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train and test splits
        """
        return train_test_split(texts, labels, 
                              test_size=test_size, 
                              random_state=random_state, 
                              stratify=labels)



class ResultsHandler:
    def __init__(self, base_path: str = os.path.join(PROJECT_ROOT, "results/evaluations")):
        """
        Initialize ResultsHandler
        Args:
            base_path (str, optional): Base path for results. Defaults to "eval".
        """
        self.base_path = base_path
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories for storing results"""
        directories = [
            "classical_ml",
            "lstm",
            "mlp",
            "transformer",
            "zero_shot",
        ]
        for dir_name in directories:
            os.makedirs(os.path.join(self.base_path, dir_name), exist_ok=True)

    def save_results(self, results: Dict[str, Any], model_name: str, category: str):
        """
        Save classification results
        Args:
            results (Dict[str, Any]): Results to save
            model_name (str): Name of the model
            category (str): Category of results
        """
        path = os.path.join(self.base_path, f"{category}", f"{model_name}.json")
        with open(path, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"Results saved to {path}")

    def plot_confusion_matrix(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            labels: List[str], 
                            model_name: str, 
                            category: str):
        """
        Plot and save confusion matrix
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Label names
            model_name (str): Name of the model
            category (str): Category of results
        """
        plt.figure(figsize=(12, 10))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        path = os.path.join(self.base_path, category, 
                           f"{model_name}_confusion_matrix.png")
        plt.savefig(path, bbox_inches='tight')
        plt.close()
        print(f"Confusion matrix saved to {path}")

    def generate_report(self, 
                       y_true: np.ndarray, 
                       y_pred: np.ndarray, 
                       labels: List[str], 
                       model_name: str, 
                       category: str) -> Dict[str, Any]:
        """
        Generate and save classification report
        Args:
            y_true (np.ndarray): True labels
            y_pred (np.ndarray): Predicted labels
            labels (List[str]): Label names
            model_name (str): Name of the model
            category (str): Category of results
        Returns:
            Dict[str, Any]: Classification report
        """
        report = classification_report(y_true, y_pred, target_names=labels, output_dict=True)
        
        path = os.path.join(self.base_path, category, 
                           f"{model_name}_classification_report.json")
        with open(path, 'w') as f:
            json.dump(report, f, indent=4)
        
        print(f"Classification report saved to {path}")
        return report

    def save_dataset_stats(self, texts: np.ndarray, labels: np.ndarray):
        """
        Save dataset statistics
        Args:
            texts (np.ndarray): Array of texts
            labels (np.ndarray): Array of labels
        """
        stats = {
            "total_samples": len(texts),
            "language_distribution": pd.Series(labels).value_counts().to_dict(),
            "avg_text_length": float(np.mean([len(text) for text in texts])),
            "min_text_length": int(min(len(text) for text in texts)),
            "max_text_length": int(max(len(text) for text in texts)),
            "unique_languages": len(np.unique(labels))
        }
        
        # Save statistics
        path = os.path.join(self.base_path, "dataset_statistics.json")
        with open(path, 'w') as f:
            json.dump(stats, f, indent=4)
        
        # Plot language distribution
        plt.figure(figsize=(15, 8))
        ax = pd.Series(labels).value_counts().plot(kind='bar')
        plt.title('Language Distribution in Dataset')
        plt.xlabel('Language')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Add value labels on top of each bar
        for i in ax.containers:
            ax.bar_label(i)
            
        plt.savefig(os.path.join(self.base_path, "language_distribution.png"), 
                    bbox_inches='tight')
        plt.close()
        
        print(f"Dataset statistics saved to {self.base_path}")

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
        self.min_delta = float(min_delta)  # Ensure min_delta is a float
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
    def __init__(
        self, 
        texts: List[str], 
        labels: List[str], 
        max_length: int = 100, 
        vocab: Optional[Dict[str, int]] = None,
        label_encoder: Optional[LabelEncoder] = None  # Add this parameter
    ):
        self.texts = texts
        self.max_length = max_length
        
        # Use existing vocabulary or create new
        if vocab is None:
            self.vocab = self._create_vocabulary(texts)
        else:
            self.vocab = vocab
        
        # Use existing label encoder or create new
        if label_encoder is None:
            self.label_encoder = LabelEncoder()
            self.labels = self.label_encoder.fit_transform(labels)
        else:
            self.label_encoder = label_encoder
            self.labels = self.label_encoder.transform(labels)  # Transform, not fit_transform
        
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


