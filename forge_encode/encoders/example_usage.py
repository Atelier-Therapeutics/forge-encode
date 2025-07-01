#!/usr/bin/env python3
"""
Example usage of the SMILES encoder.
This script demonstrates how to encode SMILES strings to latent vectors.
"""

import os
import sys
import numpy as np

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

from forge_encode.encoders.encode_smiles import SMILESEncoder, encode_smiles


def example_single_smiles():
    """Example of encoding a single SMILES string."""
    print("=== Single SMILES Encoding Example ===")
    
    # Example SMILES
    smiles = "CCO"  # ethanol
    
    # You need to provide actual paths to your trained model and vocabulary
    model_path = "path/to/your/model.ckpt.50000"  # Replace with actual path
    vocab_path = "path/to/your/vocab.txt"  # Replace with actual path
    
    try:
        # Method 1: Using the convenience function
        latent_vector = encode_smiles(smiles, model_path, vocab_path)
        if latent_vector is not None:
            print(f"SMILES: {smiles}")
            print(f"Latent vector shape: {latent_vector.shape}")
            print(f"Latent vector: {latent_vector}")
        else:
            print(f"Failed to encode SMILES: {smiles}")
            
        # Method 2: Using the SMILESEncoder class
        encoder = SMILESEncoder(model_path, vocab_path)
        latent_vector = encoder.encode_single(smiles)
        if latent_vector is not None:
            print(f"Encoded using class method - shape: {latent_vector.shape}")
            
    except FileNotFoundError as e:
        print(f"Model or vocabulary file not found: {e}")
        print("Please update the paths to point to your actual trained model and vocabulary files.")


def example_batch_smiles():
    """Example of encoding multiple SMILES strings."""
    print("\n=== Batch SMILES Encoding Example ===")
    
    # Example SMILES list
    smiles_list = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
        "CCN(CC)CC",  # triethylamine
        "c1ccc2c(c1)ccnc2",  # quinoline
        "CC(C)(C)OC(=O)N[C@@H](CC1=CC=CC=C1)C(=O)O",  # Boc-phenylalanine
    ]
    
    # You need to provide actual paths to your trained model and vocabulary
    model_path = "path/to/your/model.ckpt.50000"  # Replace with actual path
    vocab_path = "path/to/your/vocab.txt"  # Replace with actual path
    
    try:
        # Method 1: Using the convenience function
        latent_vectors = encode_smiles(smiles_list, model_path, vocab_path, batch_size=4)
        print(f"Successfully encoded {len(latent_vectors)} out of {len(smiles_list)} SMILES")
        
        for i, (smiles, vector) in enumerate(zip(smiles_list, latent_vectors)):
            if vector is not None:
                print(f"{i+1}. {smiles}: {vector.shape}")
            else:
                print(f"{i+1}. {smiles}: Failed to encode")
                
        # Method 2: Using the SMILESEncoder class
        encoder = SMILESEncoder(model_path, vocab_path)
        latent_vectors = encoder.encode_batch(smiles_list, batch_size=4)
        print(f"\nEncoded using class method - {len(latent_vectors)} vectors")
        
        # Show some statistics
        if latent_vectors:
            vectors_array = np.array(latent_vectors)
            print(f"Mean vector: {np.mean(vectors_array, axis=0)}")
            print(f"Std vector: {np.std(vectors_array, axis=0)}")
            print(f"Min values: {np.min(vectors_array, axis=0)}")
            print(f"Max values: {np.max(vectors_array, axis=0)}")
            
    except FileNotFoundError as e:
        print(f"Model or vocabulary file not found: {e}")
        print("Please update the paths to point to your actual trained model and vocabulary files.")


def example_save_vectors():
    """Example of saving encoded vectors to a file."""
    print("\n=== Save Vectors Example ===")
    
    smiles_list = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
    ]
    
    model_path = "path/to/your/model.ckpt.50000"  # Replace with actual path
    vocab_path = "path/to/your/vocab.txt"  # Replace with actual path
    output_file = "encoded_vectors.npz"  # Output file
    
    try:
        # Encode SMILES
        latent_vectors = encode_smiles(smiles_list, model_path, vocab_path)
        
        if latent_vectors:
            # Save as numpy array
            vectors_array = np.array(latent_vectors)
            np.savez(output_file, 
                    vectors=vectors_array, 
                    smiles=smiles_list,
                    latent_size=vectors_array.shape[1])
            
            print(f"Saved {len(latent_vectors)} vectors to {output_file}")
            print(f"Vector shape: {vectors_array.shape}")
            
            # Load and verify
            loaded_data = np.load(output_file)
            print(f"Loaded vectors shape: {loaded_data['vectors'].shape}")
            print(f"Loaded SMILES: {loaded_data['smiles']}")
            print(f"Latent size: {loaded_data['latent_size']}")
            
    except FileNotFoundError as e:
        print(f"Model or vocabulary file not found: {e}")
        print("Please update the paths to point to your actual trained model and vocabulary files.")


def example_command_line_equivalent():
    """Example showing how to replicate command-line functionality in Python."""
    print("\n=== Command-line Equivalent Example ===")
    
    smiles_list = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol
        "c1ccccc1",  # benzene
    ]
    
    model_path = "path/to/your/model.ckpt.50000"  # Replace with actual path
    vocab_path = "path/to/your/vocab.txt"  # Replace with actual path
    output_file = "results.json"  # Output file
    
    try:
        # This replicates the command-line functionality
        encoder = SMILESEncoder(model_path, vocab_path)
        latent_vectors = encoder.encode_batch(smiles_list, batch_size=32)
        
        # Create results dictionary (same format as command-line output)
        results = {
            'smiles': smiles_list,
            'latent_vectors': [vec.tolist() for vec in latent_vectors],
            'latent_size': encoder.model_params['latent_size'],
            'model_parameters': encoder.model_params,
            'total_processed': len(smiles_list),
            'successful_encodings': len(latent_vectors)
        }
        
        # Save to JSON (same as command-line --output)
        import json
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Results saved to {output_file}")
        print(f"Successfully encoded {len(latent_vectors)} out of {len(smiles_list)} SMILES")
        print(f"Model parameters: {encoder.model_params}")
        
    except FileNotFoundError as e:
        print(f"Model or vocabulary file not found: {e}")
        print("Please update the paths to point to your actual trained model and vocabulary files.")


def main():
    """Run all examples."""
    print("SMILES Encoder Examples")
    print("=" * 50)
    
    # Check if CUDA is available
    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Run examples
    example_single_smiles()
    example_batch_smiles()
    example_save_vectors()
    example_command_line_equivalent()
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your own model:")
    print("1. Update the model_path to point to your trained model checkpoint")
    print("2. Update the vocab_path to point to your vocabulary file")
    print("3. The model architecture parameters are automatically inferred from the checkpoint")


if __name__ == "__main__":
    main() 