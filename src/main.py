# main.py
from src.utils import DataLoader, ResultsHandler
from src.classical_ml import ClassicalMLModels
from src.deep_learning import LSTMModel, MLPModel, train_model
from src.zero_shot import GoogleTranslateDetector, LLMDetector
import torch
import pandas as pd

def main():
    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Download and load dataset
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    
    if texts is None or labels is None:
        print("Failed to load dataset. Exiting.")
        return

    # Save dataset statistics
    results_handler.save_dataset_stats(texts, labels)
    
    # Prepare data
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)

    # Classical ML
    classical_models = ClassicalMLModels()
    classical_results = classical_models.train_all_models(X_train, y_train)
    
    for model_name in classical_models.models.keys():
        predictions = classical_models.predict(model_name, X_test)
        report = results_handler.generate_report(y_test, predictions, 
                                              list(set(labels)), model_name, 
                                              "classical_ml")
        results_handler.plot_confusion_matrix(y_test, predictions, 
                                           list(set(labels)), model_name, 
                                           "classical_ml")

    # Deep Learning
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Train LSTM
    lstm_model = LSTMModel(vocab_size=10000, embedding_dim=100, 
                          hidden_dim=256, output_dim=len(set(labels)))
    # ... Training code for LSTM ...

    # Train MLP
    mlp_model = MLPModel(input_dim=10000, hidden_dim=256, 
                        output_dim=len(set(labels)))
    # ... Training code for MLP ...

    # Zero-shot approaches
    google_detector = GoogleTranslateDetector()
    llm_detector = LLMDetector(api_key="your-api-key")

    # Test zero-shot approaches on a subset
    test_subset = X_test[:100]  # Using only 100 samples for API testing
    google_results = []
    llm_results = []
    
    for text in test_subset:
        google_result = google_detector.detect_language(text)
        llm_result = llm_detector.detect_language(text)
        google_results.append(google_result)
        llm_results.append(llm_result)

    # Save zero-shot results
    results_handler.save_results(google_results, "google_translate", "zero_shot")
    results_handler.save_results(llm_results, "llm", "zero_shot")

if __name__ == "__main__":
    main()
