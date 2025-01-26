import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.mlp import run_mlp
from src.train.lstm import run_lstm
from src.train.classical_ml import run_classical_ml
from src.train.transformer import run_transformer


if __name__ == "__main__":
    print("Training Classical ML Models...")
    ml_success = run_classical_ml()
    
    print("Training MLP...")
    mlp_success = run_mlp()
    
    print("Training LSTM...")
    lstm_success = run_lstm()
    
    print("Training Transformer Model...")
    tr_success = run_transformer()