#!/usr/bin/env python3
"""
Test script to demonstrate the new encoding format with identifier and UUID-based keys.
"""

import json
import tempfile
import os
import sys

# Add the current directory to the path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from encode_smiles import main, get_machine_date_seed

def test_new_format():
    """Test the new encoding format."""
    
    # Create a temporary file with test SMILES
    test_smiles = """CCO
CC(C)O
CCOC
invalid_smiles_here
CC(C)(C)O"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(test_smiles)
        input_file = f.name
    
    # Create temporary output file
    output_file = tempfile.mktemp(suffix='.json')
    
    try:
        # Test with ATLX identifier
        print("Testing with ATLX identifier...")
        import sys
        sys.argv = [
            'encode_smiles.py',
            '--model', 'path/to/model.ckpt',  # This will fail, but we can see the argument parsing
            '--vocab', 'path/to/vocab.txt',
            '--input', input_file,
            '--output', output_file,
            '--identifier', 'ATLX',
            '--seed', '42'
        ]
        
        # We can't actually run the encoding without a real model, but we can test the argument parsing
        print("Command line arguments would be:")
        print(" ".join(sys.argv[1:]))
        
        # Test the auto-seed generation
        print(f"\nAuto-generated seed from machine+date: {get_machine_date_seed()}")
        
        # Show expected output format
        print("\nExpected output format:")
        expected_output = {
            "ATLX-a1b2c3d4e5f6": {
                "smiles": "CCO",
                "latent_vector": [0.1, 0.2, 0.3, ...],  # 32-dimensional vector
                "model_path": "model.ckpt",
                "model_parameters": {
                    "latent_size": 32,
                    "hidden_size": 250,
                    "embed_size": 250,
                    # ... other parameters
                }
            },
            "ATLX-b2c3d4e5f6g7": {
                "smiles": "CC(C)O",
                "latent_vector": [0.4, 0.5, 0.6, ...],
                "model_path": "model.ckpt",
                "model_parameters": {
                    "latent_size": 32,
                    "hidden_size": 250,
                    "embed_size": 250,
                    # ... other parameters
                }
            },
            "ATLX-c3d4e5f6g7h8": {
                "smiles": "invalid_smiles_here",
                "latent_vector": None,
                "model_path": "model.ckpt",
                "model_parameters": {
                    "latent_size": 32,
                    "hidden_size": 250,
                    "embed_size": 250,
                    # ... other parameters
                }
            }
        }
        
        print(json.dumps(expected_output, indent=2))
        
        print("\nKey features:")
        print("1. Keys are combined identifiers: ATLX + 12-character UUID")
        print("2. Values contain SMILES, latent vector, model path, and parameters")
        print("3. Failed encodings have latent_vector set to None")
        print("4. Model path shows just the filename")
        print("5. UUIDs are random but reproducible with --seed")
        print("6. Auto-generated seeds based on machine ID + date for reproducibility")
        print("7. Manual seed override still available with --seed argument")
        
    finally:
        # Clean up temporary files
        if os.path.exists(input_file):
            os.unlink(input_file)
        if os.path.exists(output_file):
            os.unlink(output_file)

if __name__ == "__main__":
    test_new_format() 