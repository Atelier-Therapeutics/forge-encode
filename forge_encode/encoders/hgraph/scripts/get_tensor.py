#!/usr/bin/env python3
"""
Script for generating tensors from molecules using HGraph encoding.

This script can be run from the command line or imported as a module.
"""

import argparse
import sys
import random
import math
import pickle
import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from multiprocessing import Pool
from functools import partial
import rdkit.Chem as Chem
from tqdm import tqdm
import torch
import networkx as nx

# Add the parent directory to the path so we can import from forge_encode
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent))

try:
    import forge_encode
    # Import your encoding modules here when they're ready
    # from forge_encode.encoders import MolecularEncoder
    from forge_encode.encoders.hgraph.mol_graph import MolGraph
    from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
    from forge_encode.encoders.hgraph.nnutils import create_pad_tensor
    # Import the add function from mol_graph
    from forge_encode.encoders.hgraph.mol_graph import add
except ImportError as e:
    print(f"Error importing forge_encode: {e}")
    sys.exit(1)

### Helper functions
def to_numpy(tensor_data):
    """Convert tensor data to numpy arrays"""
    if isinstance(tensor_data, tuple):
        return tuple(to_numpy(x) for x in tensor_data)
    elif hasattr(tensor_data, 'cpu'):
        return tensor_data.cpu().numpy()
    elif hasattr(tensor_data, 'numpy'):
        return tensor_data.numpy()
    else:
        return tensor_data

def canonicalize_tautomer(smiles, debug=False):
    """Convert SMILES to canonical tautomer and then kekulize"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        
        # Try multiple approaches to handle tautomers
        approaches = [
            # Approach 1: Direct kekulization
            lambda: Chem.MolToSmiles(mol, kekuleSmiles=True),
            
            # Approach 2: Kekulize then convert to SMILES
            lambda: Chem.MolToSmiles(Chem.Kekulize(mol), kekuleSmiles=True),
            
            # Approach 3: Use RWMol and kekulize
            lambda: Chem.MolToSmiles(Chem.Kekulize(Chem.RWMol(mol)), kekuleSmiles=True),
            
            # Approach 4: Try with different aromaticity perception
            lambda: Chem.MolToSmiles(Chem.Kekulize(mol, clearAromaticFlags=True), kekuleSmiles=True)
        ]
        
        for i, approach in enumerate(approaches):
            try:
                result = approach()
                if result:  # Accept any valid result, even if identical to original
                    if debug:
                        print(f"Approach {i+1} succeeded for {smiles}")
                    return result
            except:
                continue
        
        # If all approaches fail, return None
        return None
        
    except Exception as e:
        if debug:
            print(f"Error in tautomer canonicalization: {e}")
        return None

def vocab_hgraph(smiles):
    """Encode a molecule using the HGraph encoder and return vocabulary items"""
    hmol = MolGraph(smiles)
    vocab = set()

    # Get the vocabulary of atom clusters
    for node, attr in hmol.mol_tree.nodes(data=True):
        smiles = attr['smiles']
        vocab.add(attr['label'])
        for i, s in attr['inter_label']:
            vocab.add((smiles, s))
    
    return vocab

def tensorize_mol_batch(mol_batch, vocab):
    """Tensorize a batch of molecules"""
    try:
        x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab, show_progress=False)
        return to_numpy(x)
    except Exception as e:
        print(f"Error tensorizing batch: {e}")
        return None

def tensorize_worker_batch(worker_data):
    """Tensorize a batch of molecules with batch ID labels in progress bars"""
    try:
        mol_batch = worker_data.get('molecules', [])
        vocab = worker_data.get('vocab')
        batch_id = worker_data.get('batch_id', 0)
        mode = 'single'

        if vocab is None:
            print(f"Error: Vocabulary is None for batch {batch_id}")
            return None

        if mode == 'single':
            # Disable internal progress bar to avoid conflicts with multiprocessing
            x = MolGraph.tensorize(mol_batch, vocab, common_atom_vocab, show_progress=False)
            result = to_numpy(x)
            return result
        elif mode == 'pair':
            x, y = zip(*mol_batch)
            x = MolGraph.tensorize(x, vocab, common_atom_vocab, show_progress=False)
            y = MolGraph.tensorize(y, vocab, common_atom_vocab, show_progress=False)
            return (x, y)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        
    except Exception as e:
        batch_id = worker_data.get('batch_id', 0)
        print(f"Error tensorizing batch {batch_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def tensorize_graph_with_batch_id(graph_batch, vocab, batch_id, graph_type):
    """Tensorize graphs with batch ID in progress bar"""
    fnode, fmess = [None],[(0,0,0,0)]
    agraph, bgraph = [[]], [[]]
    scope = []
    edge_dict = {}
    all_G = []

    # Use position batch_id+3 for graph tensorization (after preprocessing and MolGraph building)
    for bid, G in enumerate(tqdm(graph_batch, desc=f"Batch {batch_id} - Tensorizing {graph_type}", leave=False, position=batch_id+3)):
        offset = len(fnode)
        scope.append((offset, len(G)))
        G = nx.convert_node_labels_to_integers(G, first_label=offset)
        all_G.append(G)
        fnode.extend([None for v in G.nodes])

        for v, attr in G.nodes(data='label'):
            G.nodes[v]['batch_id'] = bid
            fnode[v] = vocab[attr]
            agraph.append([])
        
        for u, v, attr in G.edges(data='label'):
            if type(attr) is tuple:
                fmess.append((u, v, attr[0], attr[1]))
            else:
                fmess.append((u, v, attr, 0))
            edge_dict[(u, v)] = eid = len(edge_dict) + 1
            G[u][v]['mess_idx'] = eid
            agraph[v].append(eid)
            bgraph.append([])

        for u, v in G.edges:
            eid = edge_dict[(u, v)]
            for w in G.predecessors(u):
                if w == v: continue
                bgraph[eid].append(edge_dict[(w, u)])
    
    fnode[0] = fnode[1]
    fnode = torch.IntTensor(fnode)
    fmess = torch.IntTensor(fmess)
    agraph = create_pad_tensor(agraph)
    bgraph = create_pad_tensor(bgraph)

    return (fnode, fmess, agraph, bgraph, scope), nx.union_all(all_G)

def process(batch_data, encoding_type, vocab: PairVocab = None):
    """Process a batch of molecules and generate tensors"""
    failed_molecules = []
    successful_molecules = []
    successful_tensors = []

    molecules = batch_data.get('molecules', [])
    batch_id = batch_data.get('batch_id', 0)
    
    batch_desc = f"Batch {batch_id}"
    with tqdm(molecules, desc=batch_desc, leave=False, position=batch_id+1) as pbar:
        # Pre-process all molecules in the batch
        valid_molecules = []
        for s in pbar:
            pbar.set_postfix_str(f"Processing: {s[:20]}...")

            try:
                # First check if its a valid SMILES string with RDkit
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    failed_molecules.append((s, "Invalid SMILES"))
                    continue

                # Handle tautomers by canonicalizing first
                canonical_smiles = canonicalize_tautomer(s)
                if canonical_smiles is None:
                    failed_molecules.append((s, "Tautomer canonicalization failed"))
                    continue

                valid_molecules.append((s, canonical_smiles))
                    
            except Exception as e:
                failed_molecules.append((s, str(e)))
                continue
        
        # Now tensorize the entire batch at once
        if valid_molecules and encoding_type == "hgraph_tensor":
            if vocab is None:
                # Mark all valid molecules as failed
                for original_smiles, canonical_smiles in valid_molecules:
                    failed_molecules.append((original_smiles, "Vocabulary not provided"))
            else:
                try:
                    # Extract canonical SMILES for batch tensorization
                    canonical_smiles_batch = [canonical_smiles for _, canonical_smiles in valid_molecules]
                    
                    # Tensorize the entire batch at once
                    tensor_data = MolGraph.tensorize(canonical_smiles_batch, vocab, common_atom_vocab, show_progress=False)
                    tensor_numpy = to_numpy(tensor_data)
                    
                    # Add successful molecules and tensors
                    for original_smiles, canonical_smiles in valid_molecules:
                        successful_molecules.append(canonical_smiles)
                    
                    # Store the batch tensor
                    successful_tensors.append(tensor_numpy)
                    
                except Exception as e:
                    # If batch tensorization fails, mark all molecules as failed
                    for original_smiles, canonical_smiles in valid_molecules:
                        failed_molecules.append((original_smiles, f"Batch tensorization failed: {str(e)}"))
                        
    return successful_tensors, failed_molecules, successful_molecules

def process_large_batch(batch_data, encoding_type, vocab: PairVocab = None):
    """Process a large batch of molecules efficiently"""
    failed_molecules = []
    successful_molecules = []
    successful_tensors = []

    molecules = batch_data.get('molecules', [])
    batch_id = batch_data.get('batch_id', 0)
    
    # Pre-process all molecules in the batch with progress tracking to ensure valid smiles and tautomers
    # make sure the fragments are in vocab
    # remove molecules that don't meet the criteria
    valid_molecules = []
    with tqdm(molecules, desc=f"Batch {batch_id} - Preprocessing", leave=False, position=batch_id+1) as pbar:
        for s in pbar:
            pbar.set_postfix_str(f"Processing: {s[:20]}...")
            try:
                # First check if its a valid SMILES string with RDkit
                mol = Chem.MolFromSmiles(s)
                if mol is None:
                    failed_molecules.append((s, "Invalid SMILES"))
                    continue

                # Handle tautomers by canonicalizing first
                canonical_smiles = canonicalize_tautomer(s)
                if canonical_smiles is None:
                    failed_molecules.append((s, "Tautomer canonicalization failed"))
                    continue

                # Check if the molecule fragments are in the vocabulary
                mol_vocab = vocab_hgraph(canonical_smiles)
                if not mol_vocab.issubset(set(vocab.vocab)):
                    failed_molecules.append((s, "Fragment not in vocabulary"))
                    continue

                valid_molecules.append(canonical_smiles)
                    
            except Exception as e:
                failed_molecules.append((s, str(e)))
                continue
    
    # Now tensorize the entire batch at once
    if valid_molecules and encoding_type == "hgraph_tensor":
        if vocab is None:
            # Mark all valid molecules as failed
            for canonical_smiles in valid_molecules:
                failed_molecules.append((canonical_smiles, "Vocabulary not provided"))
        else:
            try:
                # Create a new worker data structure for tensorization
                worker_data = {
                    'molecules': valid_molecules,
                    'vocab': vocab,
                    'batch_id': batch_id
                }
                
                # Use custom tensorization with batch ID labels
                tensor_data = tensorize_worker_batch(worker_data)
                if tensor_data is not None:
                    successful_tensors.append(tensor_data)
                    successful_molecules.extend(valid_molecules)
                else:
                    # If tensorization failed, mark all molecules as failed
                    for canonical_smiles in valid_molecules:
                        failed_molecules.append((canonical_smiles, "Tensorization failed"))
                        
            except Exception as e:
                # If batch tensorization fails, mark all molecules as failed
                for canonical_smiles in valid_molecules:
                    failed_molecules.append((canonical_smiles, f"Batch tensorization failed: {str(e)}"))
                    
    return successful_tensors

def tensorize_worker_function(args):
    """Worker function for tensorization"""
    batch_data, vocab_file, encoding_type = args
    try:
        # Load vocabulary in each worker process
        with open(vocab_file, 'r') as f:
            vocab_lines = [x.strip("\r\n ").split() for x in f]
        worker_vocab = PairVocab(vocab_lines, cuda=False)
        
        # Process the batch
        return process_large_batch(batch_data, encoding_type, worker_vocab)
    except Exception as e:
        print(f"Tensorization worker error: {e}")
        import traceback
        traceback.print_exc()
        return []

def simple_worker_function(smiles_list):
    """Simple worker function that just validates SMILES and returns valid ones"""
    valid_smiles = []
    for smiles in smiles_list:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = canonicalize_tautomer(smiles)
                if canonical_smiles is not None:
                    valid_smiles.append(canonical_smiles)
        except:
            continue
    return valid_smiles

def vocab_filter_worker_function(smiles_list):
    """Worker function for vocabulary filtering"""
    filtered_smiles = []
    for smiles in smiles_list:
        try:
            mol_vocab = vocab_hgraph(smiles)
            # We'll check against vocab later in the main process
            filtered_smiles.append((smiles, mol_vocab))
        except:
            continue
    return filtered_smiles

def encode_molecules(
    input_file: str,
    vocab_file: str,
    output_file: str,
    ncpu: int,
    encoding_type: str,
    **kwargs
) -> None:
    """
    Generate tensors from molecules using HGraph encoding
    
    Args:
        input_file: Path to input file containing molecular data
        vocab_file: Path to vocabulary file
        output_file: Path to output file for tensor data
        ncpu: Number of CPU cores to use
        encoding_type: Type of encoding to use
        **kwargs: Additional arguments
    """
    print(f"Getting tensors for molecules from {input_file} using {vocab_file} as vocabulary")
    
    with open(input_file, 'r') as f:
        data = [mol.strip() for line in f for mol in line.split()[:1]]  # Take only first column
    data = list(set(data))

    random.shuffle(data)

    print(f"Found {len(data)} unique molecules")

    # Load vocab from file
    print(f"Loading vocabulary from {vocab_file}")
    with open(vocab_file, 'r') as f:
        vocab_lines = [x.strip("\r\n ").split() for x in f]
    vocab_data = PairVocab(vocab_lines, cuda=False)
    
    print(f"Loaded vocabulary with {len(vocab_data.vocab)} items")

    # Use multiprocessing for preprocessing only (SMILES validation and canonicalization)
    if ncpu > 1 and len(data) > 100:
        print(f"Using multiprocessing for preprocessing with {ncpu} workers")
        
        # Split data into chunks for preprocessing
        chunk_size = max(1, len(data) // ncpu)
        data_chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]
        
        print(f"Processing {len(data)} molecules in {len(data_chunks)} chunks for preprocessing")
        
        # Use multiprocessing for preprocessing only
        with Pool(ncpu) as pool:
            try:
                # Process chunks with timeout
                timeout = 30  # 30 seconds timeout for testing
                print(f"Starting preprocessing with timeout: {timeout} seconds")
                
                async_results = pool.map_async(simple_worker_function, data_chunks)
                chunk_results = async_results.get(timeout=timeout)
                
                # Combine results
                valid_molecules = []
                for chunk in chunk_results:
                    valid_molecules.extend(chunk)
                
                print(f"Preprocessing completed: {len(valid_molecules)} valid molecules out of {len(data)}")
                
            except Exception as e:
                print(f"Error in preprocessing multiprocessing: {e}")
                print("Falling back to single-threaded preprocessing...")
                # Fallback to single-threaded preprocessing
                valid_molecules = []
                for smiles in tqdm(data, desc="Preprocessing (fallback)"):
                    try:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol is not None:
                            canonical_smiles = canonicalize_tautomer(smiles)
                            if canonical_smiles is not None:
                                valid_molecules.append(canonical_smiles)
                    except:
                        continue
    else:
        # Single-threaded preprocessing
        print("Using single-threaded preprocessing")
        valid_molecules = []
        for smiles in tqdm(data, desc="Preprocessing"):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    canonical_smiles = canonicalize_tautomer(smiles)
                    if canonical_smiles is not None:
                        valid_molecules.append(canonical_smiles)
            except:
                continue
    
    # Filter molecules that are in vocabulary using multiprocessing
    print("Filtering molecules by vocabulary...")
    if ncpu > 1 and len(valid_molecules) > 100:
        print(f"Using multiprocessing for vocabulary filtering with {ncpu} workers")
        
        # Split valid molecules into chunks for vocabulary filtering
        chunk_size = max(1, len(valid_molecules) // ncpu)
        vocab_chunks = [valid_molecules[i:i + chunk_size] for i in range(0, len(valid_molecules), chunk_size)]
        
        print(f"Processing {len(valid_molecules)} molecules in {len(vocab_chunks)} chunks for vocabulary filtering")
        
        # Use multiprocessing for vocabulary filtering
        with Pool(ncpu) as pool:
            try:
                # Process chunks with timeout
                timeout = 30  # 30 seconds timeout for vocabulary filtering
                print(f"Starting vocabulary filtering with timeout: {timeout} seconds")
                
                async_results = pool.map_async(vocab_filter_worker_function, vocab_chunks)
                vocab_results = async_results.get(timeout=timeout)
                
                # Combine and filter results
                final_molecules = []
                vocab_set = set(vocab_data.vocab)
                
                for chunk_result in tqdm(vocab_results, desc="Combining vocabulary results"):
                    for smiles, mol_vocab in chunk_result:
                        if mol_vocab.issubset(vocab_set):
                            final_molecules.append(smiles)
                
                print(f"Vocabulary filtering completed: {len(final_molecules)} valid molecules out of {len(valid_molecules)}")
                
            except Exception as e:
                print(f"Error in vocabulary filtering multiprocessing: {e}")
                print("Falling back to single-threaded vocabulary filtering...")
                # Fallback to single-threaded vocabulary filtering
                final_molecules = []
                for smiles in tqdm(valid_molecules, desc="Vocabulary filtering (fallback)"):
                    try:
                        mol_vocab = vocab_hgraph(smiles)
                        if mol_vocab.issubset(set(vocab_data.vocab)):
                            final_molecules.append(smiles)
                    except:
                        continue
    else:
        # Single-threaded vocabulary filtering
        print("Using single-threaded vocabulary filtering")
        final_molecules = []
        for smiles in tqdm(valid_molecules, desc="Vocabulary filtering"):
            try:
                mol_vocab = vocab_hgraph(smiles)
                if mol_vocab.issubset(set(vocab_data.vocab)):
                    final_molecules.append(smiles)
            except:
                continue
    
    print(f"Final valid molecules: {len(final_molecules)} out of {len(valid_molecules)}")
    
    # Process final molecules using multiprocessing for tensorization
    if final_molecules and encoding_type == "hgraph_tensor":
        print("Tensorizing molecules...")
        try:
            # Use larger batch sizes for multiprocessing
            batch_size = max(50, len(final_molecules) // (ncpu * 2))  # Ensure we have enough batches for all CPUs
            batches = []
            
            for i in range(0, len(final_molecules), batch_size):
                batch = final_molecules[i:i + batch_size]
                batches.append({
                    'molecules': batch,
                    'batch_id': i // batch_size
                })
            
            print(f"Processing {len(final_molecules)} molecules in {len(batches)} batches (batch size: {batch_size})")
            
            if ncpu > 1 and len(batches) > 1:
                print(f"Using multiprocessing for tensorization with {ncpu} workers")
                
                # Use multiprocessing for tensorization
                with Pool(ncpu) as pool:
                    try:
                        # Process batches with timeout
                        timeout = 30  # 30 seconds timeout for tensorization
                        print(f"Starting tensorization with timeout: {timeout} seconds")
                        
                        # Prepare arguments for worker function
                        worker_args = [(batch_data, vocab_file, encoding_type) for batch_data in batches]
                        
                        async_results = pool.map_async(tensorize_worker_function, worker_args)
                        batch_results = async_results.get(timeout=timeout)
                        
                        # Combine results
                        all_tensors = []
                        for result in tqdm(batch_results, desc="Combining tensorization results"):
                            if result:
                                all_tensors.extend(result)
                        
                        print(f"Generated {len(all_tensors)} tensors")
                        
                    except Exception as e:
                        print(f"Error in tensorization multiprocessing: {e}")
                        print("Falling back to single-threaded tensorization...")
                        # Fallback to single-threaded tensorization
                        all_tensors = []
                        for batch_data in tqdm(batches, desc="Tensorization (fallback)"):
                            result = process_large_batch(batch_data, encoding_type, vocab_data)
                            if result:
                                all_tensors.extend(result)
            else:
                # Single-threaded tensorization
                print("Using single-threaded tensorization")
                all_tensors = []
                for batch_data in tqdm(batches, desc="Tensorization"):
                    result = process_large_batch(batch_data, encoding_type, vocab_data)
                    if result:
                        all_tensors.extend(result)
            
            print(f"Generated {len(all_tensors)} tensors")
            
            # Save results
            base_output = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
            tensor_file = f"{base_output}_tensors.pkl"
            print(f"Saving tensors to {tensor_file}...")
            with open(tensor_file, 'wb') as f:
                pickle.dump(all_tensors, f)
            
            print(f"Tensors saved to {tensor_file}")
            print("Tensor generation completed successfully!")
            
        except Exception as e:
            print(f"Error during tensorization: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        print("No valid molecules found for tensorization")
        sys.exit(1)


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate tensors from molecules using HGraph encoding"
    )
    parser.add_argument(
        "input_file",
        help="Input file containing molecular data (SMILES, SDF, etc.)"
    )
    parser.add_argument(
        "vocab_file",
        help="Vocabulary file for encoded data"
    )
    parser.add_argument(
        "output_file",
        help="Output file for tensor data"
    )
    parser.add_argument(
        "--encoding-type",
        default="hgraph_tensor",
        choices=["hgraph_tensor"],
        help="Type of encoding to use (default: hgraph_tensor)"
    )
    parser.add_argument(
        "--ncpu",
        type=int,
        default=32,
        help="Number of CPU cores to use for parallel processing (default: 32)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"forge-encode {forge_encode.__version__}"
    )
    
    args = parser.parse_args()
    
    try:
        encode_molecules(
            input_file=args.input_file,
            vocab_file=args.vocab_file,
            output_file=args.output_file,
            ncpu=args.ncpu,
            encoding_type=args.encoding_type
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 