"""
qprofiler : Quantum Circuit Performance Profiler

A hybrid C++/Python benchmarking toolkit for state-vector quantum simulations

Modules :
qprofiler_core : compiled C++ extension (Pybind11)

Quick Start :
>>> from qprofiler import run_sweep
>>> results = run_sweep(q_min=4, q_max=24, depth=5)
"""

try:
    from .qprofiler_core import Simulator, current_rss_kb, peak_rss_kb, openmp_enabled
    _CPP_AVAILABLE = True
except:
    _CPP_AVAILABLE = False
    import warnings
    warnings.warn(
        "qprofiler_core C++ extension not found."
        "Run: cmake -B build && cmake --build build",
        ImportWarning,
        stacklevel=2
    )

from .benchmark import run_sweep, profile_gate_sequence
from .visualize import generate_report

__all__ = [
    "Simulator",
    "current_rss_kb",
    "peak_rss_kb",
    "openmp_enabled",
    "run_sweep",
    "profile_gate_sequence",
    "generate_report",
]