import sys
import os
import yaml  # Import the yaml library
from typing import Dict, List, Any, Tuple
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import joblib
from src.utils import get_path

class ClassicalMLModels:
    def __init__(self, models_dir: str = get_path("results", "models", "classical_ml")):
        """
        Initialize ClassicalMLModels
        Args:
            models_dir (str): Directory to save trained models
        """
        self.models: Dict[str, Pipeline] = {}
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Load configurations from configs.yaml
        self.configs = self._load_configs()
        self._initialize_models()

    def _load_configs(self) -> Dict[str, Any]:
        """
        Load configurations from configs.yaml
        Returns:
            Dict[str, Any]: Configurations for classical_ml
        """
        config_path = get_path("configs.yaml")
        with open(config_path, 'r') as file:
            configs = yaml.safe_load(file)
        return configs.get("classical_ml", {})  # Extract the 'classical_ml' section

    def _initialize_models(self):
        """Initialize all classical ML models with their respective pipelines using configurations from configs.yaml"""
        # Vectorizer configurations
        unigram_vec = CountVectorizer(
            ngram_range=tuple(self.configs.get("unigram_ngram_range", (1, 1))),
            max_features=self.configs.get("unigram_max_features", 50000),
            analyzer=self.configs.get("unigram_analyzer", "char")
        )
        
        bigram_vec = CountVectorizer(
            ngram_range=tuple(self.configs.get("bigram_ngram_range", (2, 2))),
            max_features=self.configs.get("bigram_max_features", 50000),
            analyzer=self.configs.get("bigram_analyzer", "char")
        )
        
        tfidf_vec = TfidfVectorizer(
            ngram_range=tuple(self.configs.get("tfidf_ngram_range", (1, 3))),
            max_features=self.configs.get("tfidf_max_features", 50000),
            analyzer=self.configs.get("tfidf_analyzer", "char")
        )

        # Model configurations
        self.models = {
            'nb_unigram': Pipeline([
                ('vectorizer', unigram_vec),
                ('classifier', MultinomialNB())
            ]),
            'nb_bigram': Pipeline([
                ('vectorizer', bigram_vec),
                ('classifier', MultinomialNB())
            ]),
            'nb_tfidf': Pipeline([
                ('vectorizer', tfidf_vec),
                ('classifier', MultinomialNB())
            ]),
            'svm_unigram': Pipeline([
                ('vectorizer', unigram_vec),
                ('classifier', LinearSVC(
                    random_state=self.configs.get("random_state", 42),
                    max_iter=self.configs.get("svm_max_iter", 1000)
                ))
            ]),
            'svm_tfidf': Pipeline([
                ('vectorizer', tfidf_vec),
                ('classifier', LinearSVC(
                    random_state=self.configs.get("random_state", 42),
                    max_iter=self.configs.get("svm_max_iter", 1000)
                ))
            ]),
            'rf_tfidf': Pipeline([
                ('vectorizer', tfidf_vec),
                ('classifier', RandomForestClassifier(
                    n_estimators=self.configs.get("rf_n_estimators", 100),
                    random_state=self.configs.get("random_state", 42)
                ))
            ]),
            'lr_tfidf': Pipeline([
                ('vectorizer', tfidf_vec),
                ('classifier', LogisticRegression(
                    random_state=self.configs.get("random_state", 42),
                    max_iter=self.configs.get("lr_max_iter", 1000)
                ))
            ])
        }

    def train_model(self, 
                   model_name: str, 
                   X_train: np.ndarray, 
                   y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train a specific model
        Args:
            model_name (str): Name of the model to train
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
        Returns:
            Dict[str, Any]: Training results
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        try:
            print(f"\nTraining {model_name}...")
            pipeline = self.models[model_name]
            pipeline.fit(X_train, y_train)
            
            # Save the trained model
            model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
            joblib.dump(pipeline, model_path)
            
            return {
                "status": "success",
                "model_path": model_path
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }

    def predict(self, 
               model_name: str, 
               X: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make predictions using a specific model
        Args:
            model_name (str): Name of the model to use
            X (np.ndarray): Features to predict
        Returns:
            Tuple[np.ndarray, Dict[str, Any]]: Predictions and metadata
        """
        if model_name not in self.models:
            return None, {"error": f"Model {model_name} not found"}
        
        try:
            pipeline = self.models[model_name]
            predictions = pipeline.predict(X)
            
            return predictions, {"status": "success"}
        except Exception as e:
            return None, {"status": "error", "error": str(e)}

    def train_all_models(self, 
                        X_train: np.ndarray, 
                        y_train: np.ndarray, 
                        X_val: np.ndarray = None, 
                        y_val: np.ndarray = None) -> Dict[str, Any]:
        """
        Train all classical ML models
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training labels
            X_val (np.ndarray, optional): Validation features
            y_val (np.ndarray, optional): Validation labels
        Returns:
            Dict[str, Any]: Training results for all models
        """
        results = {}
        
        for model_name in tqdm(self.models.keys(), desc="Training models"):
            # Train the model
            train_result = self.train_model(model_name, X_train, y_train)
            
            # Validate if validation data is provided
            val_metrics = {}
            if X_val is not None and y_val is not None:
                predictions, pred_meta = self.predict(model_name, X_val)
                if predictions is not None:
                    val_metrics["validation_accuracy"] = accuracy_score(y_val, predictions)
            
            results[model_name] = {
                "training_results": train_result,
                "validation_metrics": val_metrics
            }
        
        return results

    def load_model(self, model_name: str) -> bool:
        """
        Load a trained model from disk
        Args:
            model_name (str): Name of the model to load
        Returns:
            bool: Success status
        """
        model_path = os.path.join(self.models_dir, f"{model_name}.joblib")
        try:
            self.models[model_name] = joblib.load(model_path)
            return True
        except Exception as e:
            print(f"Error loading model {model_name}: {e}")
            return False

    def get_feature_importance(self, 
                             model_name: str, 
                             top_n: int = 20) -> Dict[str, Any]:
        """
        Get feature importance for applicable models
        Args:
            model_name (str): Name of the model
            top_n (int): Number of top features to return
        Returns:
            Dict[str, Any]: Feature importance information
        """
        if model_name not in self.models:
            return {"error": f"Model {model_name} not found"}
        
        try:
            pipeline = self.models[model_name]
            vectorizer = pipeline.named_steps['vectorizer']
            classifier = pipeline.named_steps['classifier']
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get importance scores based on model type
            if isinstance(classifier, RandomForestClassifier):
                importance_scores = classifier.feature_importances_
            elif isinstance(classifier, LinearSVC) or isinstance(classifier, LogisticRegression):
                importance_scores = np.abs(classifier.coef_).mean(axis=0)
            else:
                return {"error": "Feature importance not available for this model type"}
            
            # Sort features by importance
            features_importance = list(zip(feature_names, importance_scores))
            features_importance.sort(key=lambda x: x[1], reverse=True)
            
            return {
                "top_features": features_importance[:top_n],
                "model_type": classifier.__class__.__name__
            }
        except Exception as e:
            return {"error": str(e)}