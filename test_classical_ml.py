# test_classical_ml.py
from src.utils import DataLoader, ResultsHandler
from src.classical_ml import ClassicalMLModels
import numpy as np

def test_classical_ml():
    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    classical_models = ClassicalMLModels()
    
    # Download and load dataset
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    
    if texts is None or labels is None:
        print("Failed to load dataset")
        return False
    
    # Prepare data
    X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
    
    # Train all models
    training_results = classical_models.train_all_models(
        X_train, y_train, X_test, y_test
    )
    
    # Test each model and save results
    for model_name in classical_models.models.keys():
        print(f"\nTesting {model_name}...")
        
        # Make predictions
        predictions, pred_meta = classical_models.predict(model_name, X_test)
        
        if predictions is not None:
            # Generate and save results
            report = results_handler.generate_report(
                y_test, predictions, 
                list(np.unique(labels)), 
                model_name, 
                "classical_ml"
            )
            
            # Plot confusion matrix
            results_handler.plot_confusion_matrix(
                y_test, predictions,
                list(np.unique(labels)),
                model_name,
                "classical_ml"
            )
            
            # Get feature importance for applicable models
            importance = classical_models.get_feature_importance(model_name)
            if "error" not in importance:
                results_handler.save_results(
                    importance,
                    f"{model_name}_feature_importance",
                    "classical_ml"
                )
    
    # Save overall training results
    results_handler.save_results(
        training_results,
        "training_results",
        "classical_ml"
    )
    
    return True

if __name__ == "__main__":
    success = test_classical_ml()
    print("Test completed successfully!" if success else "Test failed!")
