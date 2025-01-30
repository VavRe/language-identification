
# Language Identification Project

This project implements and compares various approaches for language identification, ranging from classical machine learning methods to deep learning architectures and zero-shot learning with LLMs.

## Project Structure

```
├── configs.yaml           # Configuration file for all model parameters
├── dataset/              # Contains the training and testing data
├── results/              # Stores all evaluation results and trained models
├── src/                  # Source code directory
└── requirements.txt      # Project dependencies
```

## Approaches Implemented

### 1. Classical Machine Learning
- Logistic Regression with TF-IDF
- Naive Bayes (Unigram, Bigram, TF-IDF)
- Random Forest with TF-IDF
- SVM (Unigram and TF-IDF)

### 2. Deep Learning
- Transformer-based models (XLM-RoBERTa)
- LSTM (Long Short-Term Memory)
- MLP (Multi-Layer Perceptron)

### 3. Statistical Methods
- Markov Chain
- Character Frequency Analysis
- N-gram Language Models (Unigram, Trigram)

### 4. Literature-Based Methods
- Cavnar-Trenkle N-gram-based approach
- Dunning's likelihood ratio test

### 5. Zero-Shot Learning
- LLM based zero shot language identification

## Getting Started

### Prerequisites
```bash
pip install -r requirements.txt
```

### Configuration
Modify `configs.yaml` to set hyperparameters and model configurations.

### Training Models

#### Training All Models
```bash
python src/main.py
```

#### Training Specific Models
```bash
# For Classical ML
python src/train/classical_ml.py

# For Deep Learning
python src/train/transformer.py
python src/train/lstm.py
python src/train/mlp.py

# For Statistical Methods
python src/train/statistical.py

# For Literature Methods
python src/train/literature.py

# For Zero-Shot
python src/run_zero_shot.py
```

### Monitoring Training
For deep learning models, training progress can be monitored using TensorBoard:
```bash
tensorboard --logdir results/models/{model_name}/runs
```

## Results
All evaluation results are stored in `results/evaluations/` including:
- Classification reports
- Confusion matrices
- Feature importance (for applicable models)
- Training metrics

## Model Artifacts
Trained models are saved in `results/models/` organized by approach:
- Classical ML models (.joblib files)
- Deep Learning models (.pt files)
- Statistical models (.pkl files)

## Project Features
- Comprehensive comparison of different language identification approaches
- Modular code structure for easy extension
- Detailed evaluation metrics and visualizations
- TensorBoard integration for deep learning models
- Configuration-based hyperparameter management
- Structured result storage and organization

