"""Tests for __init__ module."""

import importlib.metadata
from unittest.mock import patch


def test_version_from_package():
    """Test version is loaded from package metadata."""
    from chuk_mcp_solver import __version__

    # Should either be a valid version or unknown
    assert isinstance(__version__, str)
    assert len(__version__) > 0


def test_version_fallback_on_package_not_found():
    """Test version fallback when package not found."""
    # Mock the version lookup to raise PackageNotFoundError
    with patch("importlib.metadata.version") as mock_version:
        mock_version.side_effect = importlib.metadata.PackageNotFoundError()

        # Force reimport to trigger the exception handler
        import chuk_mcp_solver

        importlib.reload(chuk_mcp_solver)

        # Should fall back to unknown version
        assert chuk_mcp_solver.__version__ == "0.0.0+unknown"
