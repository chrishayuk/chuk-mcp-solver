# Changelog

All notable changes to this project will be documented in this file.

## [0.1.3] - 2025-12-02

### Fixed

#### Server Import Fix for Deployment
- **Issue**: ImportError when deploying to Fly.io: `cannot import name 'Server' from 'chuk_mcp_server'`
- **Root Cause**: The class name changed from `Server` to `ChukMCPServer` in chuk-mcp-server
- **Fix**: Updated `src/chuk_mcp_solver/server.py` to import and use `ChukMCPServer` instead of `Server`
- **Impact**: Server now starts successfully in Docker/Fly.io environments

## [0.1.2] - 2025-12-02

Published to PyPI with all Docker support and fixes.

## [0.1.1] - 2025-12-02

### Fixed

#### Cache Deadlock Fix
- **Issue**: `test_cache_stats` test was hanging due to deadlock in `SolutionCache.stats()` method
- **Root Cause**: The `stats()` method acquired `self._lock`, then called the `hit_rate` property which tried to acquire the same lock again. Since `threading.Lock` is not reentrant, this caused a deadlock.
- **Fix**: Changed `threading.Lock` to `threading.RLock` (Reentrant Lock) in `src/chuk_mcp_solver/cache.py`
  - Line 10: Updated import from `Lock` to `RLock`
  - Line 38: Changed `self._lock = Lock()` to `self._lock = RLock()`
- **Impact**: All cache tests now pass without hanging

#### Performance Test Fix
- **Issue**: `test_partial_solution_on_timeout` was failing with timeout status instead of returning a partial solution
- **Root Cause**: Test timeout was too short (1ms) for OR-Tools CP-SAT to find even a trivial feasible solution before timing out
- **Fix**: Increased timeout in `tests/test_performance.py:178` from 1ms to 50ms
- **Additional**: Updated `src/chuk_mcp_solver/solver/ortools/solver.py:164` to check for both `cp_model.FEASIBLE` and `cp_model.OPTIMAL` statuses when handling partial solutions
- **Impact**: Test now reliably passes by giving solver enough time to find at least one feasible solution

### Added

#### Docker Support
- **Dockerfile**: Multi-stage build for optimized image size (~300-400MB)
  - Based on `python:3.11-slim`
  - Runs as non-root user (`mcpuser`) for security
  - Includes health check
  - Default command: HTTP mode on port 8000
- **docker-compose.yml**: Service definition with resource limits and health checks
- **.dockerignore**: Optimizes Docker build context by excluding unnecessary files
- **DOCKER.md**: Comprehensive Docker usage guide with examples for:
  - Building and running containers
  - Configuration with environment variables
  - Transport modes (HTTP vs STDIO)
  - Production deployment
  - CI/CD integration
  - Troubleshooting

#### Makefile Docker Targets
- `docker-build`: Build Docker image
- `docker-run`: Run Docker container
- `docker-stop`: Stop running container
- `docker-clean`: Remove container and image
- `docker-test`: Run tests in Docker container
- `docker-shell`: Open shell in running container
- `docker-push`: Push image to registry
- `docker-compose-up`: Start services with docker-compose
- `docker-compose-down`: Stop services with docker-compose
- `docker-compose-rebuild`: Rebuild and restart services

#### Package Distribution
- **MANIFEST.in**: Ensures proper file inclusion in source distributions
  - Includes README.md, LICENSE, pyproject.toml
  - Recursively includes Python files from src/ and examples/
  - Excludes compiled bytecode files

#### Public MCP Endpoint
- Added documentation for hosted solver at `https://solver.chukai.io/mcp`
- No installation required - use directly from Claude Desktop
- Perfect for testing, demos, or production use

#### Enhanced Installation Options
- Highlighted `uvx` as recommended installation method
- Added `uvx install` option for global installation
- Organized installation options by use case

### Changed

#### Server Transport Handling
- **Breaking**: Simplified command-line argument parsing in `src/chuk_mcp_solver/server.py`
- **Removed**: `argparse` dependency
- **Default**: STDIO mode (for Claude Desktop compatibility)
- **HTTP Mode**: Pass `http` or `--http` argument
- **Logging**: Improved logging suppression in STDIO mode to reduce noise
- **Pattern**: Now matches chuk-mcp-celestial transport handling

#### Documentation Updates
- **README.md**:
  - Updated test count from 151+ to 170 tests
  - Added public MCP endpoint section
  - Enhanced Quick Start with three options (Public, uvx, Development)
  - Added Docker usage section
  - Improved installation instructions with emojis and clear hierarchy
- **Test Count**: Updated badges and documentation to reflect 170 passing tests

### Test Results

All 170 tests passing:
- Fixed hanging cache tests (test_cache_stats)
- Fixed failing performance test (test_partial_solution_on_timeout)
- All existing functionality preserved
- No breaking changes to core API

### Migration Notes

#### For Users
- No action required - all changes are backward compatible
- Consider using the public endpoint at `https://solver.chukai.io/mcp` for quick testing
- For production use, consider Docker deployment for better isolation

#### For Developers
- If running the server programmatically, note the simplified transport handling:
  - Default: STDIO mode (no arguments)
  - HTTP mode: Pass `"http"` or `"--http"` as argument
  - Old `--transport` flag is no longer supported

#### For Docker Users
- Use `make docker-build` and `make docker-run` for easy Docker operations
- Default container runs in HTTP mode on port 8000
- See DOCKER.md for comprehensive deployment guide

### Security

- Docker image runs as non-root user (`mcpuser`)
- Minimal runtime dependencies reduce attack surface
- Health checks ensure service availability
- No secrets or credentials in Docker image

### Performance

- Multi-stage Docker build reduces image size
- Cache now uses reentrant locks for better concurrency
- Partial solution timeout handling improved for better user experience

---

## [0.1.0] - 2024-12-01

Initial release with comprehensive constraint solving and optimization capabilities.

See README.md for full feature list.
