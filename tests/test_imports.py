"""Tests for module imports and package structure."""


def test_import_main_module():
    """Test importing main module."""
    import chuk_mcp_solver

    assert hasattr(chuk_mcp_solver, "__version__")


def test_import_models():
    """Test importing models module."""
    from chuk_mcp_solver import models

    assert hasattr(models, "SolveConstraintModelRequest")
    assert hasattr(models, "SolveConstraintModelResponse")
    assert hasattr(models, "SolverStatus")


def test_import_config():
    """Test importing config module."""
    from chuk_mcp_solver import config

    assert hasattr(config, "Config")


def test_import_providers():
    """Test importing providers module."""
    from chuk_mcp_solver import providers

    assert hasattr(providers, "SolverProvider")
    assert hasattr(providers, "get_provider")


def test_import_server():
    """Test importing server module."""
    try:
        from chuk_mcp_solver import server

        assert hasattr(server, "main")
        assert hasattr(server, "solve_constraint_model")
    except ImportError as e:
        # chuk-mcp-server may not be importable in all test environments
        if "chuk_mcp_server" in str(e):
            pass  # Skip if MCP server framework not available
        else:
            raise


def test_version_format():
    """Test version string format."""
    from chuk_mcp_solver import __version__

    # Version should be either a valid version or unknown
    assert isinstance(__version__, str)
    assert len(__version__) > 0
