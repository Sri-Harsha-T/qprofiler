"""
Python level profiling of C++ quantum simulator

Measures:
- Python to C++ boundary overhead via cProfile
- Per qubit scaling: wall time, memory, throughput
- Gate level timing via C++ internal Profiler
"""

from __future__ import annotations

import cProfile
import pstats
import io
import time
import tracemalloc
from dataclasses import dataclass
from typing import List

try:
    import qprofiler_core as _core
    _CPP_OK = True
except ImportError:
    _CPP_OK = False

# Data classes

@dataclass
class SweepResult:
    """All metrics for a single (n_qubits, depth) run"""
    n_qubits: int
    depth: int
    wall_ms: float # C++ Wall time (ms)
    cpu_ms: float # C++ CPU time (ms)
    peak_rss_kb: int # C++ RSS at end of the run
    state_dim: int # 2^n_qubits
    state_bytes: int # memory for StateVec
    state_mb: float # state_bytes in MB
    gate_count: int
    throughput_mgs: float # million gate steps / second
    py_overhead_ms: float # Python call overhead (tracemalloc)
    tracemem_kb: float # peak Python heap during call

@dataclass
class GateRecord:
    """Single entry from C++ per gate profiler"""
    label : str
    wall_ms: float
    cpu_ms: float
    peak_rss_mb: int
    n_qubits: int
    state_dim: int
    gate_count: int


# Main sweep
def run_sweep(q_min: int = 4, q_max: int = 24, depth: int = 5, seed: int = 42, verbose: bool = True):
    """
    Sweep to run the C++ benchmark across a range of qubit counts

    Parameters:
    q_min, q_max : qubit range (inclusive)
    depth : circuit layer number (H + CNOT + Phase per layer)
    seed : RNG seed for reproducibility
    verbose : print progress to stdout

    Returns list of SweepResult instances, one per qubit count
    """
    if not _CPP_OK:
        raise RuntimeError("C++ extension not available. Build the project first")
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Quantum Profiler Sweep : {q_min} - {q_max} qubits, depth={depth}")
        print(f"{'='*60}")
        _print_header()

    results: List[SweepResult] = []

    for q in range(q_min, q_max + 1):
        sim = _core.Simulator(q)

        tracemalloc.start() # Measuring Python C++ overhead with tracemalloc
        py_start_t = time.perf_counter()
        raw = sim.run_circuit(depth = depth, seed = seed)
        py_end_t = time.perf_counter()
        _, trace_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        py_overhead_ms = (py_end_t - py_start_t) * 1000.0 - raw["wall_ms"]
        py_overhead_ms = max(py_overhead_ms, 0.0) # clamping to non-negative value

        r = SweepResult(
            n_qubits= raw["n_qubits"],
            depth= raw["depth"],
            wall_ms= raw["wall_ms"],
            cpu_ms= raw["cpu_ms"],
            peak_rss_kb= raw["peak_rss_kb"],
            state_dim = raw["state_dim"],
            state_bytes= raw["state_bytes"],
            state_mb= raw["state_bytes"] / (1024**2),
            gate_count = raw["gate_count"],
            throughput_mgs= raw["throughput_mgs"],
            py_overhead_ms= py_overhead_ms,
            tracemem_kb= trace_peak / 1024.0,
        )
        results.append(r)

        if verbose:
            _print_row(r)

    if verbose:
        print()

    return results

# Gate level profiling
def profile_gate_sequence(n_qubits: int = 16) -> List[GateRecord]:
    """
    Applies a fixed sequence of individual gates with per gate C++ profiling
    Returns a list of GateRecord objects (one per gate call)
    """

    if not _CPP_OK:
        raise RuntimeError("C++ extension not available")
    
    sim = _core.Simulator(n_qubits)
    pi = 3.141592653

    # Gate sequence
    sim.hadamard(0)
    sim.pauli_x(1%n_qubits)
    sim.pauli_z(2%n_qubits)
    sim.cnot(0, 1%n_qubits)
    sim.phase(3%n_qubits, pi/4.0)
    sim.hadamard(n_qubits - 1)
    sim.cnot(n_qubits -2 if n_qubits > 1 else 0, n_qubits - 1)

    raw_records = sim.get_records()
    return [
        GateRecord(
            label= r["label"],
            wall_ms = r["wall_ms"],
            cpu_ms = r["cpu_ms"],
            peak_rss_kb = r["peak_rss_kb"],
            n_qubits = r["n_qubits"],
            state_dim = r["state_dim"],
            gate_count = r["gate_count"],
        )
        for r in raw_records
    ]

# cprofiler boundary profiler
def cprofile_boundary(n_qubits: int = 18, depth: int = 3, top_n: int = 15) -> str:
    """
    Running cprofile around run_circuit to measure Python to C++ overhead
    Returns a formatted string of top-N hottest calls
    """

    if not _CPP_OK:
        raise RuntimeError("C++ extension not available")
    
    pr = cProfile.Profile()
    pr.enable()
    sim = _core.Simulator(n_qubits)
    sim.run_circuit(depth=depth)
    pr.disable()

    stream = io.StringIO()
    ps = pstats.Stats(pr, stream=stream).sort_stats("cumulative")
    ps.print_stats(top_n)
    return stream.getvalue()

# Formatting helper methods

def _print_header():
    cols = ("Qubits", "Dim", "StateMB", "Wall(ms)", "CPU(ms)",
            "RSS(kB)", "MGates/s", "PyOH(ms)")
    print(f"  {'  '.join(f'{c:>10}' for c in cols)}")
    print("  " + "-" * 92)


def _print_row(r: SweepResult):
    print(
        f"  {r.n_qubits:>10}  {r.state_dim:>10}  {r.state_mb:>10.3f}"
        f"  {r.wall_ms:>10.2f}  {r.cpu_ms:>10.2f}"
        f"  {r.peak_rss_kb:>10}  {r.throughput_mgs:>10.3f}"
        f"  {r.py_overhead_ms:>10.3f}"
    )





    