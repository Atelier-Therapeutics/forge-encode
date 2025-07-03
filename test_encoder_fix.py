#!/usr/bin/env python3
"""
Test script to verify the encoder fix works correctly.
"""

import torch
import sys
import os

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, project_root)

from forge_encode.encoders.hgraph.hgnn import HierVAE, make_cuda
from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
from forge_encode.encoders.hgraph.mol_graph import MolGraph

def test_encoder_fix():
    """Test that the encoder produces different outputs for different molecules."""
    
    # Test SMILES - these should produce different latent vectors
    test_smiles = [
        "CCO",  # ethanol
        "CC(C)O",  # isopropanol  
        "c1ccccc1",  # benzene
        "CC(=O)O",  # acetic acid
    ]
    
    print("Testing encoder fix...")
    print(f"Test SMILES: {test_smiles}")
    
    # Try to load the actual model and vocabulary
    model_path = "/deepLearnData/forge/ckpt/hgraph/training_chembl_200batch/model.ckpt.56000"
    vocab_path = "/deepLearnData/forge/data/chembl/proc_vocab.txt"
    
    try:
        # Load vocabulary
        print(f"Loading vocabulary from {vocab_path}")
        with open(vocab_path, 'r') as f:
            vocab_lines = [x.strip("\r\n ").split() for x in f]
        vocab = PairVocab(vocab_lines)
        print(f"Loaded vocabulary with {len(vocab.vocab)} items")
        
        # Load model
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location='cpu')
        model_state, _, _, _ = checkpoint
        
        # Infer model parameters
        def infer_model_parameters(model_state_dict):
            params = {}
            if 'R_mean.weight' in model_state_dict:
                params['latent_size'] = model_state_dict['R_mean.weight'].shape[0]
                params['hidden_size'] = model_state_dict['R_mean.weight'].shape[1]
            if 'encoder.E_c.0.weight' in model_state_dict:
                params['embed_size'] = model_state_dict['encoder.E_c.0.weight'].shape[1]
                params['vocab_size'] = model_state_dict['encoder.E_c.0.weight'].shape[0]
            if 'encoder.E_i.0.weight' in model_state_dict:
                params['ivocab_size'] = model_state_dict['encoder.E_i.0.weight'].shape[0]
            
            params.setdefault('rnn_type', 'LSTM')
            params.setdefault('depthT', 15)
            params.setdefault('depthG', 15)
            params.setdefault('diterT', 1)
            params.setdefault('diterG', 3)
            params.setdefault('dropout', 0.0)
            return params
        
        model_params = infer_model_parameters(model_state)
        print(f"Inferred model parameters: {model_params}")
        
        # Create args object
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
        
        args = Args(model_params, vocab)
        model = HierVAE(args)
        model.load_state_dict(model_state)
        
        # Move model to GPU
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded with latent_size: {args.latent_size}")
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simple test model...")
        
        # Fallback to simple model
        class Args:
            def __init__(self):
                self.rnn_type = 'LSTM'
                self.hidden_size = 250
                self.embed_size = 250
                self.latent_size = 32
                self.depthT = 15
                self.depthG = 15
                self.diterT = 1
                self.diterG = 3
                self.dropout = 0.0
                # Create a simple vocabulary
                self.vocab = PairVocab([("C", "C"), ("O", "O"), ("CC", "CC"), ("CO", "CO")])
                self.atom_vocab = common_atom_vocab
        
        args = Args()
        model = HierVAE(args)
        model.eval()
    
    # Test each SMILES individually
    results = []
    for i, smiles in enumerate(test_smiles):
        try:
            print(f"\nProcessing SMILES {i+1}: {smiles}")
            
            # Tensorize the molecule
            graphs, tensors, orders = MolGraph.tensorize([smiles], args.vocab, common_atom_vocab, show_progress=False)
            
            # Move tensors to device
            tree_tensors, graph_tensors = tensors = make_cuda(tensors)
            
            # Get encoder outputs
            with torch.no_grad():
                root_vecs, tree_vecs, _, graph_vecs = model.encoder(tree_tensors, graph_tensors)
                z_mean = model.R_mean(root_vecs)
                latent_vector = z_mean.cpu().numpy()[0]
                
                results.append((smiles, latent_vector))
                print(f"  Latent vector (first 10 dims): {latent_vector[:10]}")
                
        except Exception as e:
            print(f"  Error processing {smiles}: {e}")
            results.append((smiles, None))
    
    # Check if results are different
    print(f"\n{'='*50}")
    print("RESULTS ANALYSIS:")
    print(f"{'='*50}")
    
    valid_results = [r for r in results if r[1] is not None]
    
    if len(valid_results) < 2:
        print("❌ Not enough valid results to compare")
        return False
    
    # Compare all pairs of results
    all_different = True
    for i in range(len(valid_results)):
        for j in range(i+1, len(valid_results)):
            smiles1, vec1 = valid_results[i]
            smiles2, vec2 = valid_results[j]
            
            # Check if vectors are identical
            if torch.allclose(torch.tensor(vec1), torch.tensor(vec2), atol=1e-6):
                print(f"❌ IDENTICAL: {smiles1} and {smiles2} produce identical vectors")
                print(f"   Vector: {vec1[:5]}")
                all_different = False
            else:
                # Calculate difference
                diff = torch.norm(torch.tensor(vec1) - torch.tensor(vec2)).item()
                print(f"✅ DIFFERENT: {smiles1} vs {smiles2} - difference: {diff:.6f}")
    
    if all_different:
        print(f"\n🎉 SUCCESS: All molecules produce different latent vectors!")
        print("The encoder fix is working correctly.")
        return True
    else:
        print(f"\n❌ FAILURE: Some molecules produce identical latent vectors.")
        print("The encoder still has issues.")
        return False

if __name__ == "__main__":
    success = test_encoder_fix()
    sys.exit(0 if success else 1) 