#!/usr/bin/env python3
"""
Script and module for encoding SMILES strings to latent vectors using a trained HierVAE model.

This module can be used both as a command-line script and as an importable Python module.

Command-line usage:
    python encode_smiles.py --model model.ckpt --vocab vocab.txt --input smiles.txt --output results.json --identifier ATLX

The script generates a JSON dictionary where:
- Keys are combined identifiers (e.g., "ATLX-a1b2c3d4e5f6") with UUID-based suffixes
- Values contain SMILES, latent vector, model path, and model parameters
- Failed encodings have latent_vector set to None
- Seeds are auto-generated from machine ID + date for reproducibility (can be overridden with --seed)

Python usage:
    from forge_encode.encoders.encode_smiles import SMILESEncoder, encode_smiles
    
    # Using the class
    encoder = SMILESEncoder("model.ckpt", "vocab.txt")
    vectors = encoder.encode_batch(["CCO", "CC(C)O"])
    
    # Using the convenience function
    vectors = encode_smiles(["CCO", "CC(C)O"], "model.ckpt", "vocab.txt")
"""

import torch
import torch.nn as nn
import argparse
import os
import sys
import json
import numpy as np
import uuid
import random
import hashlib
import platform
from datetime import datetime
from rdkit import Chem
from tqdm import tqdm
from typing import List, Tuple, Optional, Union, Dict

# Add the project root to the Python path to enable imports
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..', '..', '..', '..'))
sys.path.insert(0, project_root)

# Now import the modules
from forge_encode.encoders.hgraph.hgnn import HierVAE, make_cuda
from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
from forge_encode.encoders.hgraph.mol_graph import MolGraph


def infer_model_parameters(model_state_dict):
    """
    Infer model architecture parameters from the model state dict.
    
    Args:
        model_state_dict: The state dict of the trained model
        
    Returns:
        dict: Dictionary containing the inferred parameters
    """
    params = {}
    
    # Infer latent_size from R_mean layer
    if 'R_mean.weight' in model_state_dict:
        params['latent_size'] = model_state_dict['R_mean.weight'].shape[0]
    
    # Infer hidden_size from R_mean layer
    if 'R_mean.weight' in model_state_dict:
        params['hidden_size'] = model_state_dict['R_mean.weight'].shape[1]
    
    # Infer embed_size from encoder embeddings
    if 'encoder.E_c.0.weight' in model_state_dict:
        params['embed_size'] = model_state_dict['encoder.E_c.0.weight'].shape[1]
    
    # Infer vocabulary sizes
    if 'encoder.E_c.0.weight' in model_state_dict:
        params['vocab_size'] = model_state_dict['encoder.E_c.0.weight'].shape[0]
    if 'encoder.E_i.0.weight' in model_state_dict:
        params['ivocab_size'] = model_state_dict['encoder.E_i.0.weight'].shape[0]
    
    # Set default values for parameters that can't be easily inferred
    params.setdefault('rnn_type', 'LSTM')
    params.setdefault('depthT', 15)
    params.setdefault('depthG', 15)
    params.setdefault('diterT', 1)
    params.setdefault('diterG', 3)
    params.setdefault('dropout', 0.0)
    
    return params


def canonicalize_smiles(smiles):
    """Canonicalize a SMILES string using RDKit."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None


class SMILESEncoder:
    """A class to encode SMILES strings to latent vectors using a trained HierVAE model."""
    
    def __init__(self, model_path: str, vocab_path: str, device: Optional[str] = None):
        """
        Initialize the SMILES encoder.
        
        Args:
            model_path: Path to the trained model checkpoint
            vocab_path: Path to the vocabulary file
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model_path = model_path
        self.vocab_path = vocab_path
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model, self.device, self.model_params = self._load_model()
        
    def _load_model(self) -> Tuple[HierVAE, str, dict]:
        """Load the trained HierVAE model and infer parameters."""
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        model_state, _, _, _ = checkpoint
        
        # Infer model parameters from state dict
        model_params = infer_model_parameters(model_state)
        print(f"Inferred model parameters: {model_params}")
        
        # Load vocabulary
        with open(self.vocab_path, 'r') as f:
            vocab = [x.strip("\r\n ").split() for x in f]
        vocab_obj = PairVocab(vocab)
        
        # Create args object with inferred parameters
        class Args:
            def __init__(self, params, vocab):
                self.rnn_type = params['rnn_type']
                self.hidden_size = params['hidden_size']
                self.embed_size = params['embed_size']
                self.latent_size = params['latent_size']
                self.depthT = params['depthT']
                self.depthG = params['depthG']
                self.diterT = params['diterT']
                self.diterG = params['diterG']
                self.dropout = params['dropout']
                self.vocab = vocab
                self.atom_vocab = common_atom_vocab
        
        args = Args(model_params, vocab_obj)
        
        # Create model
        model = HierVAE(args)
        
        # Load state dict
        model.load_state_dict(model_state)
        
        # Move to device
        device = torch.device(self.device)
        model = model.to(device)
        model.eval()
        
        return model, device, model_params
    
    def canonicalize_smiles(self, smiles: str) -> Optional[str]:
        """Canonicalize a SMILES string using RDKit."""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            return Chem.MolToSmiles(mol, canonical=True)
        except:
            return None
    
    def encode_single(self, smiles: str) -> Optional[np.ndarray]:
        """
        Encode a single SMILES string to a latent vector.
        
        Args:
            smiles: SMILES string to encode
            
        Returns:
            Latent vector as numpy array, or None if encoding failed
        """
        result = self.encode_batch([smiles])
        if result and len(result) > 0:
            return result[0]
        return None
    
    def encode_batch(self, smiles_list: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Encode a list of SMILES strings to latent vectors.
        
        Args:
            smiles_list: List of SMILES strings to encode
            batch_size: Batch size for processing
            
        Returns:
            List of latent vectors as numpy arrays
        """
        all_latent_vectors = []
        
        for i in range(0, len(smiles_list), batch_size):
            batch_smiles = smiles_list[i:i + batch_size]
            batch_vectors = self._encode_smiles_batch(batch_smiles)
            all_latent_vectors.extend(batch_vectors)
        
        return all_latent_vectors
    
    def _encode_smiles_batch(self, smiles_list: List[str]) -> List[np.ndarray]:
        """Encode a batch of SMILES strings to latent vectors."""
        # Preprocess SMILES
        valid_smiles = []
        valid_indices = []
        
        for i, smiles in enumerate(smiles_list):
            canonical = self.canonicalize_smiles(smiles)
            if canonical is not None:
                valid_smiles.append(canonical)
                valid_indices.append(i)
        
        if not valid_smiles:
            return []
        
        # Convert to tensors
        try:
            graphs, tensors, orders = MolGraph.tensorize(valid_smiles, self.model.vocab, common_atom_vocab, show_progress=False)
        except Exception as e:
            print(f"Error tensorizing molecules: {e}")
            return []
        
        # Move tensors to device
        tree_tensors, graph_tensors = tensors = make_cuda(tensors)
        
        # Encode to get latent vectors
        with torch.no_grad():
            root_vecs, tree_vecs, _, graph_vecs = self.model.encoder(tree_tensors, graph_tensors)
            # Get the mean of the latent distribution (no sampling)
            z_mean = self.model.R_mean(root_vecs)
        
        # Convert to numpy
        latent_vectors = z_mean.cpu().numpy()
        
        # Return vectors in original order
        result = [None] * len(smiles_list)
        for i, idx in enumerate(valid_indices):
            result[idx] = latent_vectors[i]
        
        return [vec for vec in result if vec is not None]


def encode_smiles(smiles: Union[str, List[str]], 
                 model_path: str, 
                 vocab_path: str, 
                 device: Optional[str] = None,
                 batch_size: int = 32) -> Union[Optional[np.ndarray], List[np.ndarray]]:
    """
    Convenience function to encode SMILES strings to latent vectors.
    
    Args:
        smiles: Single SMILES string or list of SMILES strings
        model_path: Path to the trained model checkpoint
        vocab_path: Path to the vocabulary file
        device: Device to use ('cuda', 'cpu', or None for auto-detection)
        batch_size: Batch size for processing (only used for lists)
        
    Returns:
        For single SMILES: latent vector as numpy array, or None if encoding failed
        For list of SMILES: list of latent vectors as numpy arrays
    """
    encoder = SMILESEncoder(model_path, vocab_path, device)
    
    if isinstance(smiles, str):
        return encoder.encode_single(smiles)
    else:
        return encoder.encode_batch(smiles, batch_size)


def load_model(model_path, vocab_path):
    """Load the trained HierVAE model and infer parameters."""
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location='cpu')
    model_state, _, _, _ = checkpoint
    
    # Infer model parameters from state dict
    model_params = infer_model_parameters(model_state)
    print(f"Inferred model parameters: {model_params}")
    
    # Load vocabulary
    vocab = [x.strip("\r\n ").split() for x in open(vocab_path)]
    vocab_obj = PairVocab(vocab)
    
    # Create args object with inferred parameters
    class Args:
        def __init__(self, params, vocab):
            self.rnn_type = params['rnn_type']
            self.hidden_size = params['hidden_size']
            self.embed_size = params['embed_size']
            self.latent_size = params['latent_size']
            self.depthT = params['depthT']
            self.depthG = params['depthG']
            self.diterT = params['diterT']
            self.diterG = params['diterG']
            self.dropout = params['dropout']
            self.vocab = vocab
            self.atom_vocab = common_atom_vocab
    
    args = Args(model_params, vocab_obj)
    
    # Create model
    model = HierVAE(args)
    
    # Load state dict
    model.load_state_dict(model_state)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    
    return model, device, model_params


def get_machine_date_seed() -> int:
    """Generate a seed based on machine ID and current date."""
    # Get machine-specific information
    machine_info = platform.node() + platform.machine() + platform.processor()
    
    # Get current date (YYYY-MM-DD format)
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Combine machine info and date
    combined_string = f"{machine_info}_{current_date}"
    
    # Generate hash and convert to integer seed
    hash_object = hashlib.md5(combined_string.encode())
    hash_hex = hash_object.hexdigest()
    
    # Convert first 8 characters of hash to integer (32-bit)
    seed = int(hash_hex[:8], 16)
    
    return seed

def generate_uuid_for_smiles(identifier: str) -> str:
    """Generate a UUID-based identifier for a SMILES string."""
    # Generate a random UUID and take first 12 characters for better uniqueness
    # This provides 2^48 possible combinations, ensuring no overlap between users
    uuid_str = str(uuid.uuid4()).replace('-', '')[:12]
    return f"{identifier}-{uuid_str}"

def encode_smiles_batch(model, smiles_list, device, vocab, atom_vocab):
    """Encode a batch of SMILES strings to latent vectors."""
    # Preprocess SMILES
    valid_smiles = []
    valid_indices = []
    
    for i, smiles in enumerate(smiles_list):
        canonical = canonicalize_smiles(smiles)
        if canonical is not None:
            valid_smiles.append(canonical)
            valid_indices.append(i)
    
    if not valid_smiles:
        return [], []
    
    # Convert to tensors
    try:
        graphs, tensors, orders = MolGraph.tensorize(valid_smiles, vocab, atom_vocab, show_progress=False)
    except Exception as e:
        print(f"Error tensorizing molecules: {e}")
        return [], []
    
    # Move tensors to device
    tree_tensors, graph_tensors = tensors = make_cuda(tensors)
    
    # Encode to get latent vectors
    with torch.no_grad():
        root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)
        # Get the mean of the latent distribution (no sampling)
        z_mean = model.R_mean(root_vecs)
    
    # Convert to numpy
    latent_vectors = z_mean.cpu().numpy()
    
    return latent_vectors, valid_indices


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Encode SMILES strings to latent vectors')
    parser.add_argument('--model', required=True, help='Path to trained model checkpoint')
    parser.add_argument('--vocab', required=True, help='Path to vocabulary file')
    parser.add_argument('--input', required=True, help='Input file with SMILES (one per line) or comma-separated SMILES string')
    parser.add_argument('--output', required=True, help='Output file for latent vectors (JSON format)')
    parser.add_argument('--identifier', required=True, choices=['ATLX', 'RCSB'], 
                       help='Four-letter identifier (ATLX or RCSB)')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for processing')
    parser.add_argument('--seed', type=int, default=None, 
                       help='Random seed for UUID generation (auto-generated from machine+date if not provided)')
    
    args = parser.parse_args()
    
    # Set random seed - use provided seed or generate from machine+date
    if args.seed is not None:
        seed = args.seed
        print(f"Using provided seed: {seed}")
    else:
        seed = get_machine_date_seed()
        print(f"Auto-generated seed from machine+date: {seed}")
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Check if input is a file or direct SMILES string
    if os.path.exists(args.input):
        # Read SMILES from file
        with open(args.input, 'r') as f:
            smiles_list = [line.strip() for line in f if line.strip()]
    else:
        # Treat as comma-separated SMILES string
        smiles_list = [s.strip() for s in args.input.split(',') if s.strip()]
    
    print(f"Processing {len(smiles_list)} SMILES strings...")
    
    # Load model and infer parameters
    print("Loading model and inferring parameters...")
    model, device, model_params = load_model(args.model, args.vocab)
    print(f"Model loaded on device: {device}")
    print(f"Model architecture: hidden_size={model_params['hidden_size']}, "
          f"embed_size={model_params['embed_size']}, latent_size={model_params['latent_size']}")
    
    # Process in batches and create results dictionary
    results = {}
    
    for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="Encoding batches"):
        batch_smiles = smiles_list[i:i + args.batch_size]
        
        latent_vectors, valid_indices = encode_smiles_batch(
            model, batch_smiles, device, model.encoder.vocab, common_atom_vocab
        )
        
        # Process each SMILES in the batch
        for j, smiles in enumerate(batch_smiles):
            # Generate unique ID for this SMILES
            unique_id = generate_uuid_for_smiles(args.identifier)
            
            # Check if this SMILES was successfully encoded
            if j in valid_indices:
                # Find the corresponding latent vector
                vec_idx = valid_indices.index(j)
                latent_vector = latent_vectors[vec_idx]
                
                results[unique_id] = {
                    'smiles': smiles,
                    'latent_vector': latent_vector.tolist(),
                    'model': os.path.basename(args.model),
                    'model_parameters': model_params
                }
            else:
                # Failed encoding
                results[unique_id] = {
                    'smiles': smiles,
                    'latent_vector': None,
                    'model': os.path.basename(args.model),
                    'model_parameters': model_params
                }
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    successful_count = sum(1 for v in results.values() if v['latent_vector'] is not None)
    failed_count = len(results) - successful_count
    
    print(f"Successfully encoded {successful_count} out of {len(smiles_list)} SMILES strings")
    print(f"Failed encodings: {failed_count}")
    print(f"Results saved to {args.output}")
    print(f"Latent vector shape: {model_params['latent_size']}")
    print(f"Output format: Dictionary with {len(results)} entries")


if __name__ == "__main__":
    main() 