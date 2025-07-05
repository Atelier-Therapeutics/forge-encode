from forge_encode.encoders.encode_smiles import main, get_machine_date_seed, generate_uuid_for_smiles
import random
import numpy as np
import torch
import json


def encode_smiles_to_json(smiles, model_path, vocab_path, identifier="ATLX", output_file="stdout"):
    """
    Wrapper function that mimics main() function in encode_smiles.py
    """

    # Set up the environment as main()
    seed = get_machine_date_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check device availability
    if torch.cuda.is_available():
        print(f"CUDA is available. Using GPU: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available. Using CPU for computation.")

    # load model and infer paramters
    from forge_encode.encoders.encode_smiles import load_model
    try:
        model, device, model_params = load_model(model_path, vocab_path)
        print(f"Model loaded from {model_path}")
    except Exception as e:
        print(f"Error loading model: {e}")
        return None
    
    # Process SMILES
    from forge_encode.encoders.encode_smiles import encode_smiles_batch
    from forge_encode.encoders.hgraph.vocab import common_atom_vocab

    smiles_list = [smiles] if isinstance(smiles, str) else smiles

    # process in batches
    results = []
    for i in range(0, len(smiles_list), 32):
        batch_smiles = smiles_list[i:i+32]

        latent_vectors, valid_indices = encode_smiles_batch(
            model, batch_smiles, device, model.encoder.vocab, common_atom_vocab
        )

        # Process each SMILES in the batch
        for j, smiles_str in enumerate(batch_smiles):
            unique_id = generate_uuid_for_smiles(identifier)
            molecule_entries = []

            # check if the smiles was successfully encoded
            if j in valid_indices:
                vec_idx = valid_indices.index(j)
                latent_vector = latent_vectors[vec_idx]

                molecule_entries.extend([
                    {
                        "key": "unique_id",
                        "value": unique_id,
                        "access_level": 2,
                        "description": "Unique identifier for this molecular encoding"
                    },
                    {
                        "key": "smiles",
                        "value": smiles_str,
                        "access_level": 1,
                        "description": f"Molecular SMILES string (ID: {unique_id})"
                    },
                    {
                        "key": "latent_vector",
                        "value": latent_vector.tolist(),
                        "access_level": 2,
                        "description": "Latent vector for ML training"
                    },
                    {
                        "key": "model",
                        "value": model_path.split('/')[-1],
                        "access_level": 2,
                        "description": "Model identifier"
                    },
                    {
                        "key": "model_parameters",
                        "value": model_params,
                        "access_level": 2,
                        "description": "Model parameters"
                    }
                ])
            else:
                # Failed encoding
                molecule_entries.extend([
                    {
                        "key": "unique_id",
                        "value": unique_id,
                        "access_level": 2,
                        "description": "Unique identifier for this molecular encoding"
                    },
                    {
                        "key": "smiles",
                        "value": smiles_str,
                        "access_level": 1,
                        "description": f"Molecular SMILES string (ID: {unique_id}) - Failed encoding"
                    },
                    {
                        "key": "model",
                        "value": model_path.split('/')[-1],
                        "access_level": 2,
                        "description": "Model identifier"
                    },
                    {
                        "key": "model_parameters",
                        "value": model_params,
                        "access_level": 2,
                        "description": "Model parameters"
                    }
                ])
            
            results.append(molecule_entries)
    
    return results

if __name__ == "__main__":
    results = encode_smiles_to_json(
        smiles="COc1cc2c(cc1O)C1Cc3sc(CCCCCl)cc3CN1CC2",
        model_path="ckpt/hgraph/model.ckpt",
        vocab_path="vocab/hgraph/proc_vocab.txt",
        identifier="ATLX",
        output_file="test.json"
    )
    print(results)
        
        
