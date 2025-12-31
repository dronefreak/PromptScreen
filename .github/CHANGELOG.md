# Changelog

All notable changes to this project will be documented in this file.

This project follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/)
and adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

_No unreleased changes._

---

## [0.2.0] - 2025-12-31

### Added

- Initial public release of **PromptScreen** on PyPi
- Updated SVM model with a better performance
- Bumped typing_extensions version

### Notes

- This is an **alpha release**; APIs may change
- ML-based guards have limited test coverage

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
