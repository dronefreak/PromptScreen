"""Data files bundled with the package."""

from pathlib import Path

DATA_DIR = Path(__file__).parent
RULES_DIR = DATA_DIR / "rules"

__all__ = ["DATA_DIR", "RULES_DIR"]