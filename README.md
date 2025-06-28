# Forge Encode

Various encoding and VAE schemes for molecules, data and interactions.

This package provides implementations of various encoding schemes and Variational Autoencoders (VAEs) for molecular data and interactions.

## Installation

### From PyPI (when published)
```bash
pip install forge-encode
```

### From source
```bash
git clone https://github.com/yourusername/forge-encode.git
cd forge-encode
pip install -e .
```

### Development installation
```bash
git clone https://github.com/yourusername/forge-encode.git
cd forge-encode
pip install -e ".[dev]"
```

## Quick Start

```python
import forge_encode

# Check version
print(f"Forge Encode version: {forge_encode.__version__}")

# Your code here...
```

## Project Structure

```
forge-encode/
├── forge_encode/          # Main package directory
│   ├── __init__.py       # Package initialization
│   ├── encoders/         # Encoding schemes
│   ├── vae/             # Variational Autoencoders
│   └── utils/           # Utility functions
├── tests/               # Test suite
├── setup.py            # Package setup
├── pyproject.toml      # Modern Python packaging
├── MANIFEST.in         # Package manifest
├── requirements.txt    # Development dependencies
└── README.md          # This file
```

## Development

### Running Tests
```bash
pytest
```

### Code Formatting
```bash
black forge_encode tests
```

### Type Checking
```bash
mypy forge_encode
```

### Linting
```bash
flake8 forge_encode tests
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this package in your research, please cite:

```bibtex
@software{forge_encode,
  title={Forge Encode: Various encoding and VAE schemes for molecules, data and interactions},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/forge-encode}
}
```
