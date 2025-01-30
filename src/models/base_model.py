import sys
import os

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))



from abc import ABC, abstractmethod
import pickle
import os
from src.utils import get_path

class BaseModel(ABC):
    def __init__(self, model_name: str, category:str):
        self.model_name = model_name
        self.model_dir = get_path("results", "models",category, self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)

    def save(self):
        """Save model to disk"""
        model_path = os.path.join(self.model_dir, f"{self.model_name}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, model_path: str):
        """Load model from disk"""
        with open(model_path, 'rb') as f:
            return pickle.load(f)

    @abstractmethod
    def train(self, texts, labels):
        """Train the model"""
        pass

    @abstractmethod
    def predict(self, texts):
        """Make predictions"""
        pass
