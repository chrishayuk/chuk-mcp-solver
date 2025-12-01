"""CHUK MCP Solver - General-purpose constraint and optimization solver MCP server."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("chuk-mcp-solver")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
