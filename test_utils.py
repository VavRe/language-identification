# test_utils.py
from src.utils import DataLoader, ResultsHandler

def test_data_loading():
    # Initialize components
    data_loader = DataLoader()
    results_handler = ResultsHandler()
    
    # Download and load dataset
    dataset_path = data_loader.download_dataset()
    texts, labels = data_loader.load_dataset(dataset_path)
    
    if texts is not None and labels is not None:
        # Save dataset statistics
        results_handler.save_dataset_stats(texts, labels)
        
        # Prepare data
        X_train, X_test, y_train, y_test = data_loader.prepare_data(texts, labels)
        print(f"Train set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        return True
    return False

if __name__ == "__main__":
    success = test_data_loading()
    print("Test completed successfully!" if success else "Test failed!")
