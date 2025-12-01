"""Tests for configuration management."""

import os
from unittest.mock import patch

from chuk_mcp_solver.config import Config, load_yaml_config


def test_config_default_provider():
    """Test default provider configuration."""
    assert Config.DEFAULT_PROVIDER == "ortools"


def test_config_get_provider_for_tool():
    """Test getting provider for specific tool."""
    provider = Config.get_provider_for_tool("solve_constraint_model")
    assert provider == "ortools"


def test_config_get_provider_for_unknown_tool():
    """Test getting provider for unknown tool returns default."""
    provider = Config.get_provider_for_tool("unknown_tool")
    assert provider == Config.DEFAULT_PROVIDER


@patch.dict(os.environ, {"CHUK_SOLVER_PROVIDER": "custom_provider"})
def test_config_env_var_override():
    """Test that environment variable overrides default."""
    # Need to reload the module to pick up env var
    import importlib

    from chuk_mcp_solver import config

    importlib.reload(config)

    assert config.Config.DEFAULT_PROVIDER == "custom_provider"


def test_load_yaml_config_missing_file(tmp_path):
    """Test loading config when file doesn't exist."""
    with patch("chuk_mcp_solver.config.CONFIG_PATH", tmp_path / "nonexistent.yaml"):
        config = load_yaml_config()
        assert config == {}


def test_load_yaml_config_valid_file(tmp_path):
    """Test loading valid YAML config."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text(
        "default_provider: custom\ntool_providers:\n  solve_constraint_model: special\n"
    )

    with patch("chuk_mcp_solver.config.CONFIG_PATH", config_file):
        config = load_yaml_config()
        assert config["default_provider"] == "custom"
        assert config["tool_providers"]["solve_constraint_model"] == "special"


def test_load_yaml_config_invalid_yaml(tmp_path):
    """Test loading invalid YAML returns empty dict."""
    config_file = tmp_path / "config.yaml"
    config_file.write_text("invalid: yaml: content: [")

    with patch("chuk_mcp_solver.config.CONFIG_PATH", config_file):
        config = load_yaml_config()
        assert config == {}
