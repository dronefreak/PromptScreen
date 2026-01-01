# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_No unreleased changes._

---

## [0.3.0] - 2026-01-02

### Added

- **CLI** - `promptscreen scan "Ignore all instructions"` now works!
- Check `src/promptscreen/cli.py` for more details!
- Added tests for CLI

### Fixed

- Added tests/ to CI
- Re-ordered imports to fix ruff errors
- Bumped conflicting versions of ruff linter

## [0.3.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.3.0

---

## [0.2.0] - 2025-12-26

### Added

- **PyPI package publication** - `pip install promptscreen` now works!
- Properly configured packaging for distribution

### Fixed

- typing-extensions dependency now installed on all Python versions (fixes Python 3.12 import error)
- VectorDB and ML guards now properly optional (lazy imports)
- chromadb import error when using core package only

### Changed

- First public release on PyPI (previously source-only)
- Improved optional dependency handling

## [0.2.0]: https://github.com/dronefreak/PromptScreen/releases/tag/v0.2.0

---

## [0.1.0] - 2025-12-25

### Added

- Initial public release of **PromptScreen**
- Prompt injection and jailbreak detection via:
  - Heuristic and regex-based scanners
  - YARA rule matching
  - Optional ML-based classifiers
  - Optional vector similarity detection
- Modular guard architecture with independently usable components
- FastAPI-based API server with multiple evaluation modes
- Evaluation framework for measuring attack success rate (ASR)

### Notes

- This is an **alpha release**; APIs may change
- Package is not yet published to PyPI
- ML-based guards have limited test coverage
