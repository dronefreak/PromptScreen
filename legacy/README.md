# Legacy Code (Archived)

This directory contains the original flat-structure code before refactoring.

**DO NOT USE THESE FILES.** They are kept only for reference.

## Use New Structure Instead

All functionality has been moved to `src/promptscreen/`:
- `defence/` → `src/promptscreen/defence/`
- `pipeline.py` → `src/promptscreen/evaluation/pipeline.py`
- `guards.py` → `examples/run_*.py`
- `main.py` → `examples/run_*.py`
- `api.py` → `src/promptscreen/api/server.py`

See `examples/` directory for usage.

## Deletion Plan

This directory will be deleted in v1.0 release after:
- [ ] All examples working
- [ ] All tests passing
- [ ] Package installable via pip
- [ ] Documentation complete