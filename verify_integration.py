import torch
import sys
import os

# Add src to path
sys.path.append(os.path.abspath("src"))

from experiment import ExperimentPCNet, ExperimentRandomEnsemblePC

def test_single_pcnet():
    print("Testing ExperimentPCNet...")
    config = {
        "model_name": "spatial_pcnet",
        "input_channels": 3,
        "n_mixtures": 5,
        "max_depth": 3,
        "channels_per_layer": 16,
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu"
    }
    
    exp = ExperimentPCNet(config)
    # Check if model is initialized
    if exp.model is None:
        print("❌ ExperimentPCNet model not initialized.")
        return
    
    # Init network with dummy data
    data = torch.randn(2, 3, 32, 32)
    exp.model.init_network(data)
    
    # Forward
    out = exp.model(data)
    print(f"✅ ExperimentPCNet Forward Output: {out.shape}")

def test_ensemble_pcnet():
    print("\nTesting ExperimentRandomEnsemblePC...")
    config = {
        "model_name": "ensemble_pcnet",
        "n_models": 2,
        "n_jobs": 1, # Use 1 for simple testing
        "input_channels": 3,
        "n_mixtures": 5,
        "max_depth": 3,
        "channels_per_layer": 16,
        "epochs": 1,
        "batch_size": 2,
        "device": "cpu"
    }
    
    exp = ExperimentRandomEnsemblePC(config)
    
    # Init network with dummy data
    data = torch.randn(2, 3, 32, 32)
    
    # Manual init for ensemble (usually happens in train/fit)
    exp.model.fit(data)
    
    # Forward
    out = exp.model(data)
    print(f"✅ ExperimentRandomEnsemblePC Forward Output: {out.shape}")

if __name__ == "__main__":
    os.makedirs("logs", exist_ok=True)
    test_single_pcnet()
    test_ensemble_pcnet()
