"""
Test that the package can be imported correctly.
"""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import forge_encode
    assert forge_encode.__version__ is not None


def test_version_format():
    """Test that the version is a string."""
    import forge_encode
    assert isinstance(forge_encode.__version__, str)


def test_author_info():
    """Test that author information is available."""
    import forge_encode
    assert hasattr(forge_encode, "__author__")
    assert hasattr(forge_encode, "__email__") 