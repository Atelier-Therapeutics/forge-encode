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
import multiprocessing as mp
import os
import signal
import atexit
import gc

# Configure PyTorch multiprocessing to prevent file descriptor issues
torch.multiprocessing.set_sharing_strategy('file_system')

# Additional PyTorch multiprocessing settings to prevent memory issues
os.environ['OMP_NUM_THREADS'] = '1'  # Prevent OpenMP from spawning too many threads
os.environ['MKL_NUM_THREADS'] = '1'  # Prevent MKL from spawning too many threads

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

def increase_file_descriptor_limit():
    """Try to increase file descriptor limit to prevent 'Too many open files' errors"""
    try:
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Try to increase to hard limit or a reasonable value
        target_limit = min(hard_limit, 65536)  # Cap at 65536 to be safe
        
        if soft_limit < target_limit:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target_limit, hard_limit))
            new_soft, new_hard = resource.getrlimit(resource.RLIMIT_NOFILE)
            print(f"Increased file descriptor limit from {soft_limit} to {new_soft}")
            return new_soft
        else:
            print(f"File descriptor limit already sufficient: {soft_limit}")
            return soft_limit
    except Exception as e:
        print(f"Warning: Could not increase file descriptor limit: {e}")
        return None

# Global variable to track active pools for cleanup
_active_pools = []

def get_optimal_worker_count(requested_cpus, max_workers_per_core=2):
    """Calculate optimal number of workers to prevent file descriptor issues"""
    try:
        # Get system file descriptor limit
        import resource
        soft_limit, hard_limit = resource.getrlimit(resource.RLIMIT_NOFILE)
        
        # Conservative estimate: each worker might use ~100 file descriptors
        # Leave some headroom for system processes
        max_workers_by_fd = max(1, (soft_limit // 100) - 10)
        
        # Also consider CPU count
        import multiprocessing
        cpu_count = multiprocessing.cpu_count()
        max_workers_by_cpu = cpu_count * max_workers_per_core
        
        # Take the minimum to be safe
        optimal_workers = min(requested_cpus, max_workers_by_fd, max_workers_by_cpu)
        
        if optimal_workers < requested_cpus:
            print(f"Warning: Reducing workers from {requested_cpus} to {optimal_workers} to prevent file descriptor issues")
            print(f"  File descriptor limit: {soft_limit}, CPU count: {cpu_count}")
        
        return optimal_workers
    except Exception as e:
        print(f"Warning: Could not determine optimal worker count: {e}")
        # Fallback to conservative estimate
        return min(requested_cpus, 8)

def cleanup_pools():
    """Clean up all active multiprocessing pools"""
    global _active_pools
    for pool in _active_pools:
        try:
            if pool and not pool._state == 'CLOSE':
                pool.close()
                pool.join()
        except:
            pass
    _active_pools.clear()

def signal_handler(signum, frame):
    """Handle signals to ensure proper cleanup"""
    print(f"\nReceived signal {signum}, cleaning up...")
    cleanup_pools()
    sys.exit(0)

# Register cleanup functions
atexit.register(cleanup_pools)
signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# Set multiprocessing start method
try:
    mp.set_start_method('spawn', force=True)
except RuntimeError:
    # Already set, ignore
    pass

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

def batch_tensorize_worker_function(args):
    """Worker function for batch tensorization with tqdm progress bar"""
    batch_data, vocab_file, encoding_type, worker_id = args
    try:
        from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
        from forge_encode.encoders.hgraph.mol_graph import MolGraph
        from tqdm import tqdm
        
        # Load vocabulary in worker process
        with open(vocab_file, 'r') as f:
            vocab_lines = [x.strip("\r\n ").split() for x in f]
        worker_vocab = PairVocab(vocab_lines, cuda=False)
        
        molecules = batch_data.get('molecules', [])
        batch_id = batch_data.get('batch_id', 0)
        
        # Add space for single-digit worker IDs to align progress bars
        worker_display = f" {worker_id}" if worker_id < 10 else str(worker_id)
        desc = f"Batch Tensor Worker {worker_display} (batch {batch_id}, size={len(molecules)})"
        
        # Tensorize the entire batch at once
        batch_result = MolGraph.tensorize(molecules, worker_vocab, common_atom_vocab, show_progress=False)
        
        # Convert to numpy arrays to avoid PyTorch tensor sharing issues
        numpy_result = to_numpy(batch_result)
        
        # Clean up memory
        del batch_result, worker_vocab
        gc.collect()
        
        return numpy_result
        
    except Exception as e:
        batch_id = batch_data.get('batch_id', 0)
        print(f"Error tensorizing batch {batch_id}: {e}")
        import traceback
        traceback.print_exc()
        return None

def tensorize_worker_function(args):
    """Worker function for tensorization with tqdm progress bar for each batch"""
    batch_data, vocab_file, encoding_type, worker_id = args
    try:
        from forge_encode.encoders.hgraph.vocab import PairVocab, common_atom_vocab
        from tqdm import tqdm
        with open(vocab_file, 'r') as f:
            vocab_lines = [x.strip("\r\n ").split() for x in f]
        worker_vocab = PairVocab(vocab_lines, cuda=False)
        atom_vocab = common_atom_vocab
        molecules = batch_data.get('molecules', [])
        batch_id = batch_data.get('batch_id', 0)
        # Add space for single-digit worker IDs to align progress bars
        worker_display = f" {worker_id}" if worker_id < 10 else str(worker_id)
        desc = f"Tensor Worker {worker_display} (batch {batch_id}, size={len(molecules)})"
        # Show a progress bar for this batch
        # Process the entire batch at once, but show a tqdm for per-molecule progress
        results = []
        for i, smiles in enumerate(tqdm(molecules, 
                                       desc=desc, 
                                       position=worker_id, 
                                       leave=False,
                                       mininterval=1.0,  # Update at least every 1 second
                                       maxinterval=1.0)):  # Update at most every 1 second
            try:
                tensor = MolGraph.tensorize([smiles], worker_vocab, atom_vocab, show_progress=False)
                # Convert to numpy immediately to avoid PyTorch tensor sharing issues
                numpy_tensor = to_numpy(tensor)
                results.append(numpy_tensor)
            except Exception:
                continue
        return results
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []

def simple_worker_function(args):
    """Simple worker function that just validates SMILES and returns valid ones"""
    smiles_list, worker_id = args
    from tqdm import tqdm
    valid_smiles = []
    # Add space for single-digit worker IDs to align progress bars
    worker_display = f" {worker_id}" if worker_id < 10 else str(worker_id)
    for smiles in tqdm(smiles_list, 
                      desc=f"Preprocessing Worker {worker_display}", 
                      position=worker_id, 
                      leave=False,
                      mininterval=1.0,  # Update at least every 1 second
                      maxinterval=1.0):  # Update at most every 1 second
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                canonical_smiles = canonicalize_tautomer(smiles)
                if canonical_smiles is not None:
                    valid_smiles.append(canonical_smiles)
        except:
            continue
    return valid_smiles

def vocab_filter_worker_function(args):
    """Worker function for vocabulary filtering with progress bar"""
    smiles_list, worker_id = args
    filtered_smiles = []
    from tqdm import tqdm
    
    # Add space for single-digit worker IDs to align progress bars
    worker_display = f" {worker_id}" if worker_id < 10 else str(worker_id)
    for smiles in tqdm(smiles_list, 
                      desc=f"Vocab Filter Worker {worker_display}", 
                      position=worker_id, 
                      leave=False,
                      mininterval=1.0,  # Update at least every 1 second
                      maxinterval=1.0):  # Update at most every 1 second
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
    batch_size: int = None,
    training_batch_size: int = 50,
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
        batch_size: Batch size for tensorization
        training_batch_size: Number of tensors to group into each training batch
        **kwargs: Additional arguments
    """
    print(f"Getting tensors for molecules from {input_file} using {vocab_file} as vocabulary")
    
    # Try to increase file descriptor limit
    increase_file_descriptor_limit()
    
    # Calculate optimal number of workers to prevent file descriptor issues
    optimal_ncpu = get_optimal_worker_count(ncpu)
    
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
    if optimal_ncpu > 1 and len(data) > 100:
        print(f"Using multiprocessing for preprocessing with {optimal_ncpu} workers")
        
        # Split data into exactly optimal_ncpu chunks for optimal parallelization
        num_chunks = optimal_ncpu
        chunk_size = max(1, len(data) // num_chunks)
        data_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(data)
            data_chunks.append(data[start_idx:end_idx])
        
        print(f"Processing {len(data)} molecules in {len(data_chunks)} chunks for preprocessing")
        
        # Use multiprocessing for preprocessing only
        pool = None
        try:
            pool = Pool(optimal_ncpu)
            _active_pools.append(pool)
            
            # Process chunks with timeout
            timeout = 300  # 30 seconds timeout for testing
            print(f"Starting preprocessing with timeout: {timeout} seconds")
            
            async_results = pool.map_async(simple_worker_function, [(chunk, i) for i, chunk in enumerate(data_chunks)])
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
            for smiles in tqdm(data, 
                              desc="Preprocessing (fallback)",
                              mininterval=1.0,  # Update at least every 1 second
                              maxinterval=1.0):  # Update at most every 1 second
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol is not None:
                        canonical_smiles = canonicalize_tautomer(smiles)
                        if canonical_smiles is not None:
                            valid_molecules.append(canonical_smiles)
                except:
                    continue
        finally:
            if pool:
                try:
                    pool.close()
                    pool.join()
                    if pool in _active_pools:
                        _active_pools.remove(pool)
                except:
                    pass
    else:
        # Single-threaded preprocessing
        print("Using single-threaded preprocessing")
        valid_molecules = []
        for smiles in tqdm(data, 
                          desc="Preprocessing",
                          mininterval=1.0,  # Update at least every 1 second
                          maxinterval=1.0):  # Update at most every 1 second
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
    if optimal_ncpu > 1 and len(valid_molecules) > 100:
        print(f"Using multiprocessing for vocabulary filtering with {optimal_ncpu} workers")
        
        # Split valid molecules into exactly optimal_ncpu chunks for optimal parallelization
        num_chunks = optimal_ncpu
        chunk_size = max(1, len(valid_molecules) // num_chunks)
        vocab_chunks = []
        
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < num_chunks - 1 else len(valid_molecules)
            vocab_chunks.append(valid_molecules[start_idx:end_idx])
        
        print(f"Processing {len(valid_molecules)} molecules in {len(vocab_chunks)} chunks for vocabulary filtering")
        
        # Use multiprocessing for vocabulary filtering
        pool = None
        try:
            pool = Pool(optimal_ncpu)
            _active_pools.append(pool)
            
            # Process chunks with timeout
            timeout = 600  # 600 seconds timeout for vocabulary filtering
            print(f"Starting vocabulary filtering with timeout: {timeout} seconds")
            
            async_results = pool.map_async(vocab_filter_worker_function, [(chunk, i) for i, chunk in enumerate(vocab_chunks)])
            vocab_results = async_results.get(timeout=timeout)
            
            # Combine and filter results
            final_molecules = []
            vocab_set = set(vocab_data.vocab)
            
            for chunk_result in tqdm(vocab_results, 
                                    desc="Combining vocabulary results",
                                    mininterval=1.0,  # Update at least every 1 second
                                    maxinterval=1.0):  # Update at most every 1 second
                for smiles, mol_vocab in chunk_result:
                    if mol_vocab.issubset(vocab_set):
                        final_molecules.append(smiles)
            
            print(f"Vocabulary filtering completed: {len(final_molecules)} valid molecules out of {len(valid_molecules)}")
            
        except Exception as e:
            print(f"Error in vocabulary filtering multiprocessing: {e}")
            print("Falling back to single-threaded vocabulary filtering...")
            # Fallback to single-threaded vocabulary filtering
            final_molecules = []
            for smiles in tqdm(valid_molecules, 
                              desc="Vocabulary filtering (fallback)",
                              mininterval=1.0,  # Update at least every 1 second
                              maxinterval=1.0):  # Update at most every 1 second
                try:
                    mol_vocab = vocab_hgraph(smiles)
                    if mol_vocab.issubset(set(vocab_data.vocab)):
                        final_molecules.append(smiles)
                except:
                    continue
        finally:
            if pool:
                try:
                    pool.close()
                    pool.join()
                    if pool in _active_pools:
                        _active_pools.remove(pool)
                except:
                    pass
    else:
        # Single-threaded vocabulary filtering
        print("Using single-threaded vocabulary filtering")
        final_molecules = []
        for smiles in tqdm(valid_molecules, 
                          desc="Vocabulary filtering",
                          mininterval=1.0,  # Update at least every 1 second
                          maxinterval=1.0):  # Update at most every 1 second
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
            # Group molecules into batches and tensorize each batch
            molecules_per_batch = training_batch_size  # Number of molecules per batch
            num_batches = (len(final_molecules) + molecules_per_batch - 1) // molecules_per_batch
            
            print(f"Grouping {len(final_molecules)} molecules into {num_batches} batches of ~{molecules_per_batch} molecules each...")
            
            # Create output directory
            base_output = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
            output_dir = f"{base_output}_processed"
            os.makedirs(output_dir, exist_ok=True)
            
            # Prepare batches for multiprocessing
            batch_data = []
            for i in range(num_batches):
                start_idx = i * molecules_per_batch
                end_idx = min(start_idx + molecules_per_batch, len(final_molecules))
                molecule_batch = final_molecules[start_idx:end_idx]
                batch_data.append({
                    'molecules': molecule_batch,
                    'batch_id': i
                })
            
            # Use multiprocessing for batch tensorization
            # Use only half of available CPUs for tensorization to prevent memory exhaustion
            tensorization_cpus = max(1, optimal_ncpu // 2)
            if optimal_ncpu > 1 and len(batch_data) > 1:
                print(f"Using multiprocessing for batch tensorization with {tensorization_cpus} workers (half of {optimal_ncpu} available CPUs)")
                
                # Process batches in smaller chunks to reduce memory pressure
                chunk_size = max(1, len(batch_data) // 4)  # Process in 4 chunks
                all_batches = []
                
                for chunk_idx in range(0, len(batch_data), chunk_size):
                    chunk_end = min(chunk_idx + chunk_size, len(batch_data))
                    batch_chunk = batch_data[chunk_idx:chunk_end]
                    
                    print(f"Processing chunk {chunk_idx//chunk_size + 1}/{(len(batch_data) + chunk_size - 1)//chunk_size}: batches {chunk_idx}-{chunk_end-1}")
                    
                    # Use multiprocessing for batch tensorization
                    pool = None
                    try:
                        pool = Pool(tensorization_cpus)
                        _active_pools.append(pool)
                        
                        # Process batches with timeout
                        timeout = 1800  # 30 minutes timeout for batch tensorization
                        
                        # Prepare arguments for worker function
                        worker_args = [(batch_info, vocab_file, encoding_type, i) for i, batch_info in enumerate(batch_chunk)]
                        
                        async_results = pool.map_async(batch_tensorize_worker_function, worker_args)
                        batch_results = async_results.get(timeout=timeout)
                        
                        # Filter out None results
                        chunk_batches = [result for result in batch_results if result is not None]
                        all_batches.extend(chunk_batches)
                        
                        print(f"Successfully created {len(chunk_batches)} batches in chunk")
                        
                    except Exception as e:
                        print(f"Error in batch tensorization multiprocessing for chunk: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Falling back to single-threaded batch tensorization for this chunk...")
                        # Fallback to single-threaded tensorization
                        chunk_batches = []
                        for batch_info in tqdm(batch_chunk, 
                                              desc="Batch tensorization (fallback)",
                                              mininterval=1.0,
                                              maxinterval=1.0):
                            result = batch_tensorize_worker_function((batch_info, vocab_file, encoding_type, batch_info['batch_id']))
                            if result is not None:
                                chunk_batches.append(result)
                        all_batches.extend(chunk_batches)
                    finally:
                        if pool:
                            try:
                                pool.close()
                                pool.join()
                                if pool in _active_pools:
                                    _active_pools.remove(pool)
                            except:
                                pass
                    
                    # Force garbage collection between chunks
                    gc.collect()
                
                batches = all_batches
                print(f"Successfully created {len(batches)} total batches")
            else:
                # Single-threaded batch tensorization
                print("Using single-threaded batch tensorization")
                batches = []
                for batch_info in tqdm(batch_data, 
                                      desc="Batch tensorization",
                                      mininterval=1.0,
                                      maxinterval=1.0):
                    result = batch_tensorize_worker_function((batch_info, vocab_file, encoding_type, batch_info['batch_id']))
                    if result is not None:
                        batches.append(result)
            
            # Save batches as multiple files (each file contains a list of batches)
            # DataFolder expects multiple .pkl files, each containing a list of batches
            batches_per_file = 10  # Each file will contain 10 batches
            num_files = (len(batches) + batches_per_file - 1) // batches_per_file
            
            print(f"Saving {len(batches)} batches to {num_files} files ({batches_per_file} batches per file)...")
            
            for file_idx in range(num_files):
                start_idx = file_idx * batches_per_file
                end_idx = min(start_idx + batches_per_file, len(batches))
                file_batches = batches[start_idx:end_idx]
                
                file_path = os.path.join(output_dir, f"batch_{file_idx:04d}.pkl")
                with open(file_path, 'wb') as f:
                    pickle.dump(file_batches, f)
                
                print(f"Saved file {file_idx+1}/{num_files}: {len(file_batches)} batches")
            
            print(f"Saved {len(batches)} total batches across {num_files} files")
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
        "--batch-size",
        type=int,
        help="Batch size for tensorization"
    )
    parser.add_argument(
        "--training-batch-size",
        type=int,
        default=50,
        help="Number of tensors to group into each training batch"
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
            encoding_type=args.encoding_type,
            batch_size=args.batch_size,
            training_batch_size=args.training_batch_size
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 