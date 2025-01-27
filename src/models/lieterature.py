from collections import defaultdict
import numpy as np
from typing import List, Dict, Any, Tuple
from .base_model import BaseModel

class CavnarTrenkleClassifier(BaseModel):
    def __init__(self, max_profile_size: int = 400):
        super().__init__(model_name="cavnar_trenkle", category="literature")
        self.max_profile_size = max_profile_size
        self.language_profiles = {}

    def _get_ngram_frequencies(self, text: str) -> List[Tuple[str, int]]:
        """Get ordered n-gram frequencies for a text"""
        ngrams = defaultdict(int)
        for n in range(1, 6):  # n-grams of length 1-5
            text_padded = '_' * (n-1) + text.lower() + '_' * (n-1)
            for i in range(len(text_padded)-n+1):
                ngrams[text_padded[i:i+n]] += 1
        
        # Sort by frequency, then alphabetically
        return sorted(ngrams.items(), key=lambda x: (-x[1], x[0]))[:self.max_profile_size]

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        training_stats = {'n_samples': len(texts)}
        
        for label in set(labels):
            # Combine all texts for this language
            combined_text = ' '.join(text for text, l in zip(texts, labels) if l == label)
            # Store ordered n-gram profile
            self.language_profiles[label] = self._get_ngram_frequencies(combined_text)
        
        training_stats['n_languages'] = len(self.language_profiles)
        return training_stats

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            text_profile = self._get_ngram_frequencies(text)
            text_ngrams = [ng for ng, _ in text_profile]
            
            # Calculate "out of place" measure for each language
            distances = {}
            for lang, lang_profile in self.language_profiles.items():
                lang_ngrams = [ng for ng, _ in lang_profile]
                distance = 0
                
                # Calculate distance based on ranking differences
                for i, ngram in enumerate(text_ngrams):
                    try:
                        j = lang_ngrams.index(ngram)
                        distance += abs(i - j)
                    except ValueError:
                        distance += self.max_profile_size
                
                distances[lang] = distance
            
            predictions.append(min(distances.items(), key=lambda x: x[1])[0])
        return predictions

class DunningClassifier(BaseModel):
    def __init__(self, smoothing: float = 0.5):  # Increased smoothing
        super().__init__(model_name="dunning", category="literature")
        self.language_models = {}
        self.smoothing = smoothing
        self.vocabulary = set()
        
    def _get_features(self, text: str) -> Dict[str, int]:
        """Extract more balanced features"""
        features = defaultdict(int)
        text = text.lower()
        
        # Single characters (baseline features)
        for char in text:
            features[f'c_{char}'] += 1
        
        # Character bigrams
        text_pad = '_' + text + '_'
        for i in range(len(text_pad)-1):
            features[f'bg_{text_pad[i:i+2]}'] += 1
        
        return features

    def train(self, texts: List[str], labels: List[str]) -> Dict[str, Any]:
        training_stats = {'n_samples': len(texts)}
        
        # First pass: collect all features
        all_features = set()
        lang_features = defaultdict(lambda: defaultdict(float))
        lang_totals = defaultdict(float)
        
        for text, label in zip(texts, labels):
            features = self._get_features(text)
            all_features.update(features.keys())
            for feat, count in features.items():
                lang_features[label][feat] += count
                lang_totals[label] += count
        
        self.vocabulary = all_features
        
        # Second pass: ensure all features are present for each language
        for label in set(labels):
            for feat in all_features:
                if feat not in lang_features[label]:
                    lang_features[label][feat] = 0
        
        # Calculate probabilities with proper smoothing
        for label in set(labels):
            total = lang_totals[label]
            smoothed_total = total + self.smoothing * len(all_features)
            self.language_models[label] = {
                feat: (count + self.smoothing) / smoothed_total
                for feat, count in lang_features[label].items()
            }
        
        # Add debugging information
        training_stats.update({
            'n_languages': len(self.language_models),
            'vocabulary_size': len(self.vocabulary),
            'features_per_language': {
                lang: sum(1 for v in features.values() if v > 0)
                for lang, features in lang_features.items()
            }
        })
        
        # Print feature distribution
        print("\nFeature distribution per language:")
        for lang in set(labels):
            n_active = sum(1 for v in lang_features[lang].values() if v > 0)
            print(f"{lang}: {n_active}/{len(all_features)} features active")
        
        return training_stats

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        prediction_stats = defaultdict(int)
        
        for text in texts:
            scores = {}
            features = self._get_features(text)
            
            for lang, model in self.language_models.items():
                # Calculate normalized log probability
                score = 0
                n_features = 0
                for feat, count in features.items():
                    if count > 0:
                        prob = model.get(feat, self.smoothing)
                        score += count * np.log(prob)
                        n_features += count
                
                # Normalize by number of features to avoid length bias
                scores[lang] = score / n_features if n_features > 0 else float('-inf')
            
            prediction = max(scores.items(), key=lambda x: x[1])[0]
            predictions.append(prediction)
            prediction_stats[prediction] += 1
        
        # Print prediction distribution
        print("\nPrediction distribution:")
        total_predictions = len(texts)
        for lang, count in sorted(prediction_stats.items()):
            print(f"{lang}: {count} ({count/total_predictions*100:.1f}%)")
        
        return predictions
