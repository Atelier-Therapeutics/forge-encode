#!/usr/bin/env python3
"""
Script for encoding molecules using various encoding schemes.

This script can be run from the command line or imported as a module.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from multiprocessing import Pool
from functools import partial
import rdkit.Chem as Chem
from tqdm import tqdm

# Add the parent directory to the path so we can import from forge_encode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge_encode
    # Import your encoding modules here when they're ready
    # from forge_encode.encoders import MolecularEncoder
    from forge_encode.encoders.hgraph.mol_graph import MolGraph
except ImportError as e:
    print(f"Error importing forge_encode: {e}")
    sys.exit(1)

### Helper functions
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
    """Encode a molecule using the HGraph encoder"""
    hmol = MolGraph(smiles)
    vocab = set()

    # Get the vocabulary of atom clusters
    for node, attr in hmol.mol_tree.nodes(data=True):
        smiles = attr['smiles']
        vocab.add(attr['label'])
        for i, s in attr['inter_label']:
            vocab.add((smiles, s))
    
    return vocab

def process(batch_data, encoding_type):
    """Process a batch of molecules using various encoding schemes"""
    vocab = set()
    failed_molecules = []
    successful_molecules = []

    molecules = batch_data.get('molecules', [])
    batch_id = batch_data.get('batch_id', 0)
    
    batch_desc = f"Batch {batch_id}"
    with tqdm(molecules, desc=batch_desc, leave=False, position=batch_id+1) as pbar:
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

                if encoding_type == "hgraph_vocab":
                    # Generate initial vocabulary for the molecule
                    mol_vocab = vocab_hgraph(canonical_smiles)
                    vocab.update(mol_vocab)  # Accumulate vocabulary
                    successful_molecules.append(canonical_smiles)
                elif encoding_type == "hgraph_tensor":
                    # get vocab from file based on input_file
                    vocab_file = input_file.rsplit('.', 1)[0] if '.' in input_file else input_file
                    vocab_file = vocab_file.rsplit('_', 1)[0] if '_' in vocab_file else vocab_file
                    vocab_file = f"{vocab_file}_vocab.txt"
                    with open(vocab_file, 'r') as f:
                        vocab = [x.strip("\r\n ").split() for x in f]
                    vocab = PairVocab(vocab, cuda=False)
                    # Check if the molecule clusters are in the vocabulary
                    mol_vocab = vocab_hgraph(canonical_smiles)
                    if mol_vocab is None:
                        pass
                    
                    
                else:
                    raise ValueError(f"Unsupported encoding type: {encoding_type}")
            except Exception as e:
                failed_molecules.append((s, str(e)))
                continue
    return vocab, failed_molecules, successful_molecules

def encode_molecules(
    input_file: str,
    output_file: str,
    ncpu: int = 32,
    encoding_type: str = "hgraph_vocab",
    **kwargs
) -> None:
    """
    Encode molecules from input file and save to output file(s)
    
    Args:
        input_file: Path to input file containing molecular data
        output_file: Path to output file for encoded data
        encoding_type: Type of encoding to use (fingerprint, graph, etc.)
        **kwargs: Additional arguments for the encoder
    """
    print(f"Encoding molecules from {input_file} to {output_file}")
    print(f"Using encoding type: {encoding_type}")
    
    with open(input_file, 'r') as f:
        data = [mol for line in f for mol in line.split()[:2]]
    data = list(set(data))

    print(f"Found {len(data)} unique molecules")

    # define batch size and batches based on ncpu and data size
    batch_size = len(data) // ncpu + 1
    batches = [data[i : i + batch_size] for i in range(0, len(data), batch_size)]

    
    print(f"Processing {len(data)} molecules in {len(batches)} batches....")

    # Prepare batches with encoding_type
    prepared_batches = []
    for i, batch in enumerate(batches):
        prepared_batches.append({
            'molecules': batch,
            'batch_id': i
        })
    
    # Use multiprocessing only if we have multiple batches or multiple CPUs
    if len(batches) > 1 or ncpu > 1:
        with tqdm(total=len(prepared_batches), desc="Overall Progress", position=0) as main_pbar:
            pool = Pool(ncpu)
            results = []
            try:
                for result in pool.imap(partial(process, encoding_type=encoding_type), prepared_batches):
                    results.append(result)
                    main_pbar.update(1)
            finally:
                pool.close()
                pool.join()
    else:
        # Single-threaded processing for small datasets
        results = []
        for batch_data in tqdm(prepared_batches, desc="Processing"):
            result = process(batch_data, encoding_type)
            results.append(result)

    # Process results and combine vocabularies
    combined_vocab = set()
    all_failed_molecules = []
    all_successful_molecules = []
    
    for vocab, failed_mols, successful_mols in results:
        combined_vocab.update(vocab)
        all_failed_molecules.extend(failed_mols)
        all_successful_molecules.extend(successful_mols)
    
    print(f"Successfully processed {len(all_successful_molecules)} molecules")
    print(f"Failed to process {len(all_failed_molecules)} molecules")
    print(f"Generated vocabulary with {len(combined_vocab)} unique items")
    
    # Save results to separate text files
    base_output = output_file.rsplit('.', 1)[0] if '.' in output_file else output_file
    
    # Save vocabulary
    vocab_file = f"{base_output}_vocab.txt"
    with open(vocab_file, 'w') as f:
        for item in sorted(combined_vocab):
            if isinstance(item, tuple):
                f.write(f"{item[0]}\t{item[1]}\n")
            else:
                f.write(f"{item}\n")
    
    # Save successful molecules
    success_file = f"{base_output}_successful.txt"
    with open(success_file, 'w') as f:
        for smiles in all_successful_molecules:
            f.write(f"{smiles}\n")
    
    # Save failed molecules with reasons
    failed_file = f"{base_output}_failed.txt"
    with open(failed_file, 'w') as f:
        for smiles, reason in all_failed_molecules:
            f.write(f"{smiles}\t{reason}\n")
    
    print(f"Vocabulary saved to {vocab_file}")
    print(f"Successful molecules saved to {success_file}")
    print(f"Failed molecules saved to {failed_file}")
    print("Encoding completed successfully!")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Encode molecules using various encoding schemes"
    )
    parser.add_argument(
        "input_file",
        help="Input file containing molecular data (SMILES, SDF, etc.)"
    )
    parser.add_argument(
        "output_file",
        help="Output file for encoded data"
    )
    parser.add_argument(
        "--encoding-type",
        default="hgraph_vocab",
        choices=["hgraph_vocab"],
        help="Type of encoding to use (default: hgraph_vocab)"
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
    
    # Add more arguments as needed
    # parser.add_argument("--radius", type=int, default=2, help="Fingerprint radius")
    # parser.add_argument("--n-bits", type=int, default=2048, help="Number of bits")
    
    args = parser.parse_args()
    
    try:
        encode_molecules(
            input_file=args.input_file,
            output_file=args.output_file,
            ncpu=args.ncpu,
            encoding_type=args.encoding_type
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 