"""
Forge Encode - Various encoding and VAE schemes for molecules, data and interactions.

This package provides implementations of various encoding schemes and Variational
Autoencoders (VAEs) for molecular data and interactions.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

__author__ = "Your Name"
__email__ = "your.email@example.com"

# Import main modules here
# from . import encoders
# from . import vae
# from . import utils

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    # Add your main modules here
    # "encoders",
    # "vae", 
    # "utils",
] 