#!/usr/bin/env python3
"""
Test script to verify parameter inference from model checkpoints.
"""

import torch
import os
import sys

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from forge_encode.encoders.smiles_encoder import infer_model_parameters


def test_parameter_inference():
    """Test parameter inference with a sample model checkpoint."""
    
    # Example model checkpoint path (you would need to provide a real one)
    model_path = "path/to/your/model.ckpt.50000"
    
    if not os.path.exists(model_path):
        print(f"Model checkpoint not found: {model_path}")
        print("Please provide a valid model checkpoint path to test parameter inference.")
        return
    
    try:
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state, _, _, _ = checkpoint
        
        print("Model checkpoint loaded successfully!")
        print(f"Checkpoint contains: {len(model_state)} state dict keys")
        
        # Test parameter inference
        params = infer_model_parameters(model_state)
        
        print("\nInferred parameters:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Print some key layer shapes for verification
        print("\nKey layer shapes:")
        if 'R_mean.weight' in model_state:
            print(f"  R_mean.weight: {model_state['R_mean.weight'].shape}")
        if 'R_var.weight' in model_state:
            print(f"  R_var.weight: {model_state['R_var.weight'].shape}")
        if 'encoder.E_c.0.weight' in model_state:
            print(f"  encoder.E_c.0.weight: {model_state['encoder.E_c.0.weight'].shape}")
        if 'encoder.E_i.0.weight' in model_state:
            print(f"  encoder.E_i.0.weight: {model_state['encoder.E_i.0.weight'].shape}")
        
        # Verify parameter consistency
        print("\nParameter consistency check:")
        if 'R_mean.weight' in model_state:
            inferred_latent = params['latent_size']
            inferred_hidden = params['hidden_size']
            actual_shape = model_state['R_mean.weight'].shape
            
            print(f"  R_mean layer shape: {actual_shape}")
            print(f"  Inferred latent_size: {inferred_latent} (should be {actual_shape[0]})")
            print(f"  Inferred hidden_size: {inferred_hidden} (should be {actual_shape[1]})")
            
            assert inferred_latent == actual_shape[0], f"Latent size mismatch: {inferred_latent} vs {actual_shape[0]}"
            assert inferred_hidden == actual_shape[1], f"Hidden size mismatch: {inferred_hidden} vs {actual_shape[1]}"
            print("  ✓ Parameter inference is correct!")
        
    except Exception as e:
        print(f"Error testing parameter inference: {e}")
        import traceback
        traceback.print_exc()


def create_dummy_checkpoint():
    """Create a dummy checkpoint for testing (if no real checkpoint is available)."""
    print("Creating dummy checkpoint for testing...")
    
    # Create a dummy model state dict with typical shapes
    dummy_state = {
        'R_mean.weight': torch.randn(32, 250),  # latent_size=32, hidden_size=250
        'R_mean.bias': torch.randn(32),
        'R_var.weight': torch.randn(32, 250),
        'R_var.bias': torch.randn(32),
        'encoder.E_c.0.weight': torch.randn(1000, 250),  # vocab_size=1000, embed_size=250
        'encoder.E_c.0.bias': torch.randn(1000),
        'encoder.E_i.0.weight': torch.randn(500, 250),  # ivocab_size=500, embed_size=250
        'encoder.E_i.0.bias': torch.randn(500),
    }
    
    # Create dummy checkpoint
    dummy_checkpoint = (dummy_state, {}, 0, 0.0)
    
    # Test parameter inference
    params = infer_model_parameters(dummy_state)
    
    print("Dummy checkpoint created!")
    print("Inferred parameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Verify expected values
    expected = {
        'latent_size': 32,
        'hidden_size': 250,
        'embed_size': 250,
        'vocab_size': 1000,
        'ivocab_size': 500,
        'rnn_type': 'LSTM',
        'depthT': 15,
        'depthG': 15,
        'diterT': 1,
        'diterG': 3,
        'dropout': 0.0
    }
    
    print("\nVerification:")
    for key, expected_value in expected.items():
        if key in params:
            actual_value = params[key]
            status = "✓" if actual_value == expected_value else "✗"
            print(f"  {status} {key}: {actual_value} (expected: {expected_value})")


if __name__ == "__main__":
    print("Testing Parameter Inference")
    print("=" * 40)
    
    # Try to test with real checkpoint first
    test_parameter_inference()
    
    print("\n" + "=" * 40)
    print("Testing with dummy checkpoint:")
    create_dummy_checkpoint()
    
    print("\n" + "=" * 40)
    print("Parameter inference test completed!") 