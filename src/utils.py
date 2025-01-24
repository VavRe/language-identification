# src/utils.py
import os
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
    def __init__(self, base_path: str = "eval"):
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
            "classical_ml_results",
            "deep_learning_results",
            "zero_shot_results"
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
        path = os.path.join(self.base_path, f"{category}_results", f"{model_name}_results.json")
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
        
        path = os.path.join(self.base_path, f"{category}_results", 
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
        
        path = os.path.join(self.base_path, f"{category}_results", 
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
