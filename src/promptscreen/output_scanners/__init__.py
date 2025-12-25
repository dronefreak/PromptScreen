"""Output scanning for LLM responses."""

try:
    from .scan import OutputScanner

    _has_output_scanner = True
except ImportError:
    OutputScanner = None  # type: ignore
    _has_output_scanner = False

__all__ = ["OutputScanner"]
