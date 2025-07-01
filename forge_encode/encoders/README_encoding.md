# SMILES Encoding Scripts

This directory contains scripts for encoding SMILES strings to latent vectors using a trained HierVAE model.

## Files

- `encode_smiles.py` - **Main script and module** for encoding SMILES strings (command-line and Python usage)
- `example_usage.py` - Example script demonstrating usage
- `test_parameter_inference.py` - Test script for parameter inference
- `README_encoding.md` - This file

## Prerequisites

1. A trained HierVAE model checkpoint (e.g., `model.ckpt.50000`)
2. A vocabulary file (e.g., `vocab.txt`) that was used during training
3. **No need to specify model architecture parameters** - they are automatically inferred from the checkpoint

## Usage

### 1. Command-line Script (`encode_smiles.py`)

This script is useful for batch processing SMILES strings from files or command line.

```bash
# Encode SMILES from a file (one SMILES per line)
python encode_smiles.py \
    --model path/to/your/model.ckpt.50000 \
    --vocab path/to/your/vocab.txt \
    --input smiles_file.txt \
    --output results.json \
    --batch_size 32

# Encode SMILES directly from command line
python encode_smiles.py \
    --model path/to/your/model.ckpt.50000 \
    --vocab path/to/your/vocab.txt \
    --input "CCO,CC(C)O,c1ccccc1" \
    --output results.json
```

**Arguments:**
- `--model`: Path to trained model checkpoint
- `--vocab`: Path to vocabulary file
- `--input`: Input file with SMILES (one per line) or comma-separated SMILES string
- `--output`: Output JSON file for results
- `--batch_size`: Batch size for processing (default: 32)

**Output Format:**
The script outputs a JSON file with:
```json
{
  "smiles": ["CCO", "CC(C)O", ...],
  "latent_vectors": [[0.1, 0.2, ...], [0.3, 0.4, ...], ...],
  "latent_size": 32,
  "model_parameters": {
    "hidden_size": 250,
    "embed_size": 250,
    "latent_size": 32,
    "rnn_type": "LSTM",
    ...
  },
  "total_processed": 100,
  "successful_encodings": 95
}
```

### 2. Python Module (`encode_smiles.py`)

The same file can be imported and used as a Python module.

#### Using the convenience function:

```python
from forge_encode.encoders.encode_smiles import encode_smiles

# Encode a single SMILES
smiles = "CCO"
latent_vector = encode_smiles(smiles, "model.ckpt.50000", "vocab.txt")
if latent_vector is not None:
    print(f"Latent vector shape: {latent_vector.shape}")

# Encode multiple SMILES
smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
latent_vectors = encode_smiles(smiles_list, "model.ckpt.50000", "vocab.txt", batch_size=32)
print(f"Encoded {len(latent_vectors)} molecules")
```

#### Using the SMILESEncoder class:

```python
from forge_encode.encoders.encode_smiles import SMILESEncoder

# Initialize encoder
encoder = SMILESEncoder("model.ckpt.50000", "vocab.txt", device="cuda")

# Encode single SMILES
latent_vector = encoder.encode_single("CCO")

# Encode batch of SMILES
smiles_list = ["CCO", "CC(C)O", "c1ccccc1"]
latent_vectors = encoder.encode_batch(smiles_list, batch_size=32)

# Access model parameters
print(f"Model latent size: {encoder.model_params['latent_size']}")

# Save vectors
import numpy as np
vectors_array = np.array(latent_vectors)
np.save("encoded_vectors.npy", vectors_array)
```

### 3. Example Script (`example_usage.py`)

Run the example script to see all usage patterns:

```bash
python example_usage.py
```

This will demonstrate:
- Single SMILES encoding
- Batch SMILES encoding
- Saving vectors to files
- Command-line equivalent functionality
- Error handling

## Automatic Parameter Inference

The scripts automatically infer model architecture parameters from the trained model checkpoint:

- **`latent_size`**: From `R_mean.weight.shape[0]`
- **`hidden_size`**: From `R_mean.weight.shape[1]`
- **`embed_size`**: From `encoder.E_c.0.weight.shape[1]`
- **`vocab_size`**: From `encoder.E_c.0.weight.shape[0]`
- **`ivocab_size`**: From `encoder.E_i.0.weight.shape[0]`

Parameters that can't be easily inferred use sensible defaults:
- **`rnn_type`**: 'LSTM'
- **`depthT`**, **`depthG`**: 15
- **`diterT`**, **`diterG`**: 1, 3
- **`dropout`**: 0.0

## Error Handling

The scripts handle various error cases:

1. **Invalid SMILES**: Invalid SMILES strings are skipped
2. **Vocabulary mismatch**: Molecules with fragments not in vocabulary are skipped
3. **Tensorization errors**: Molecules that fail to tensorize are skipped
4. **Model loading errors**: Clear error messages for missing files
5. **Parameter inference errors**: Graceful fallback to defaults

## Performance Tips

1. **Batch size**: Use larger batch sizes (32-64) for better GPU utilization
2. **GPU usage**: The scripts automatically use CUDA if available
3. **Memory**: For large datasets, process in smaller batches to avoid memory issues
4. **Vocabulary filtering**: Pre-filter SMILES to ensure they're in vocabulary for better performance

## Testing

Run the parameter inference test to verify everything works:

```bash
python test_parameter_inference.py
```

This will test the parameter inference with both real and dummy checkpoints.

## Troubleshooting

### Common Issues:

1. **"Model or vocabulary file not found"**
   - Check that the paths to your model checkpoint and vocabulary file are correct
   - Ensure the files exist and are readable

2. **"Error tensorizing molecules"**
   - Some molecules may have fragments not in your vocabulary
   - Try canonicalizing SMILES first
   - Check that the vocabulary file matches your training vocabulary

3. **"CUDA out of memory"**
   - Reduce the batch size
   - Process smaller batches
   - Use CPU if GPU memory is insufficient

4. **"Parameter inference failed"**
   - The script will use default parameters
   - Check that your model checkpoint is from a HierVAE model
   - Verify the checkpoint format is correct

### Getting Help:

1. Check the example script for working usage patterns
2. Verify your model checkpoint and vocabulary file are from the same training run
3. Ensure all dependencies are installed (PyTorch, RDKit, etc.)
4. Run the test script to verify parameter inference works with your checkpoint 