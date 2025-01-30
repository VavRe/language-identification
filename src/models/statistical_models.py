# src/models/statistical_models.py
from collections import defaultdict
import numpy as np
from typing import List, Dict, Any
from .base_model import BaseModel

class NgramModel(BaseModel):
    def __init__(self, n: int = 1, smoothing: float = 1e-8):
        super().__init__(model_name=f"ngram_{n}", category="statistical")
        self.n = n
        self.smoothing = smoothing
        self.language_models = {}

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        training_stats = {'n_samples': len(texts)}
        
        unique_labels = set(labels)
        for label in unique_labels:
            label_texts = [text for text, l in zip(texts, labels) if l == label]
            ngrams = defaultdict(lambda: self.smoothing)
            
            for text in label_texts:
                text = '_' * (self.n-1) + text + '_'
                for i in range(len(text)-self.n+1):
                    ngram = text[i:i+self.n]
                    ngrams[ngram] += 1
            
            total = sum(ngrams.values())
            self.language_models[label] = {k: v/total for k, v in ngrams.items()}
        
        training_stats['n_languages'] = len(unique_labels)
        return training_stats

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            scores = {}
            text = '_' * (self.n-1) + text + '_'
            
            for language, model in self.language_models.items():
                score = 0
                for i in range(len(text)-self.n+1):
                    ngram = text[i:i+self.n]
                    if ngram in model:
                        score += np.log(model[ngram] + self.smoothing)
                scores[language] = score
            
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        return predictions

class CharacterFrequency(BaseModel):
    def __init__(self, smoothing: float = 1e-10):
        super().__init__(model_name="char_freq", category="statistical")
        self.language_profiles = {}
        self.smoothing = smoothing

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        training_stats = {'n_samples': len(texts)}
        
        unique_labels = set(labels)
        for label in unique_labels:
            label_texts = [text.lower() for text, l in zip(texts, labels) if l == label]
            char_counts = defaultdict(lambda: self.smoothing)
            
            for text in label_texts:
                for char in text:
                    char_counts[char] += 1
            
            total = sum(char_counts.values())
            self.language_profiles[label] = {char: count/total 
                                           for char, count in char_counts.items()}
        
        training_stats['n_languages'] = len(unique_labels)
        return training_stats

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            text = text.lower()
            char_counts = defaultdict(lambda: self.smoothing)
            for char in text:
                char_counts[char] += 1
            
            total = sum(char_counts.values())
            text_profile = {char: count/total for char, count in char_counts.items()}
            
            scores = {}
            for language, profile in self.language_profiles.items():
                common_chars = set(text_profile.keys()) & set(profile.keys())
                numerator = sum(text_profile[char] * profile[char] for char in common_chars)
                
                denominator1 = np.sqrt(sum(v**2 for v in text_profile.values()))
                denominator2 = np.sqrt(sum(v**2 for v in profile.values()))
                
                similarity = numerator / (denominator1 * denominator2 + self.smoothing)
                scores[language] = similarity
            
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        return predictions

class MarkovChain(BaseModel):
    def __init__(self, smoothing: float = 1e-10):
        super().__init__(model_name="markov_chain", category="statistical")
        self.language_models = {}
        self.smoothing = smoothing

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        training_stats = {'n_samples': len(texts)}
        
        unique_labels = set(labels)
        for label in unique_labels:
            label_texts = [text for text, l in zip(texts, labels) if l == label]
            transition_counts = defaultdict(lambda: defaultdict(lambda: self.smoothing))
            
            for text in label_texts:
                text = '_' + text + '_'
                for i in range(len(text)-1):
                    current = text[i]
                    next_char = text[i+1]
                    transition_counts[current][next_char] += 1
            
            self.language_models[label] = {}
            for current, next_chars in transition_counts.items():
                total = sum(next_chars.values())
                self.language_models[label][current] = {next_char: count/total 
                                                      for next_char, count in next_chars.items()}
        
        training_stats['n_languages'] = len(unique_labels)
        return training_stats

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            scores = {}
            text = '_' + text + '_'
            
            for language, model in self.language_models.items():
                score = 0
                for i in range(len(text)-1):
                    current = text[i]
                    next_char = text[i+1]
                    if current in model and next_char in model[current]:
                        score += np.log(model[current][next_char] + self.smoothing)
                scores[language] = score
            
            predictions.append(max(scores.items(), key=lambda x: x[1])[0])
        return predictions

