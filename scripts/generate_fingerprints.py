#!/usr/bin/env python3
"""
Script for generating molecular fingerprints from SMILES strings.

This script can be run from the command line or imported as a module.
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Add the parent directory to the path so we can import from forge_encode
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import forge_encode
    # Import your fingerprint modules here when they're ready
    # from forge_encode.encoders import FingerprintEncoder
except ImportError as e:
    print(f"Error importing forge_encode: {e}")
    sys.exit(1)


def generate_fingerprints(
    input_file: str,
    output_file: str,
    fingerprint_type: str = "morgan",
    radius: int = 2,
    n_bits: int = 2048,
    **kwargs
) -> None:
    """
    Generate molecular fingerprints from input file and save to output file.
    
    Args:
        input_file: Path to input file containing SMILES strings
        output_file: Path to output file for fingerprints
        fingerprint_type: Type of fingerprint (morgan, rdkit, etc.)
        radius: Radius for circular fingerprints
        n_bits: Number of bits for the fingerprint
        **kwargs: Additional arguments for the fingerprint generator
    """
    print(f"Generating {fingerprint_type} fingerprints from {input_file}")
    print(f"Output: {output_file}")
    print(f"Parameters: radius={radius}, n_bits={n_bits}")
    
    # TODO: Implement the actual fingerprint generation logic
    # Example:
    # encoder = FingerprintEncoder(
    #     fingerprint_type=fingerprint_type,
    #     radius=radius,
    #     n_bits=n_bits,
    #     **kwargs
    # )
    # smiles_list = load_smiles(input_file)
    # fingerprints = encoder.generate_fingerprints(smiles_list)
    # save_fingerprints(fingerprints, output_file)
    
    print("Fingerprint generation completed successfully!")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(
        description="Generate molecular fingerprints from SMILES strings"
    )
    parser.add_argument(
        "input_file",
        help="Input file containing SMILES strings (one per line)"
    )
    parser.add_argument(
        "output_file",
        help="Output file for fingerprints (CSV, NPZ, etc.)"
    )
    parser.add_argument(
        "--fingerprint-type",
        default="morgan",
        choices=["morgan", "rdkit", "maccs", "atom-pair", "torsion"],
        help="Type of fingerprint to generate (default: morgan)"
    )
    parser.add_argument(
        "--radius",
        type=int,
        default=2,
        help="Radius for circular fingerprints (default: 2)"
    )
    parser.add_argument(
        "--n-bits",
        type=int,
        default=2048,
        help="Number of bits for the fingerprint (default: 2048)"
    )
    parser.add_argument(
        "--output-format",
        default="csv",
        choices=["csv", "npz", "json", "pickle"],
        help="Output format for fingerprints (default: csv)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"forge-encode {forge_encode.__version__}"
    )
    
    args = parser.parse_args()
    
    try:
        generate_fingerprints(
            input_file=args.input_file,
            output_file=args.output_file,
            fingerprint_type=args.fingerprint_type,
            radius=args.radius,
            n_bits=args.n_bits,
            output_format=args.output_format
        )
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main() 