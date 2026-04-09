# qprofiler

A high-performance profiler for quantum circuit simulations built in C++ with Python bindings. It answers the questions regarding the time and memory consumption when a quantum circuit is simulated and what overhead costs exist, by measuring the wall-clock/ CPU time, memory consumption, gate throughput and it is enabled with OpenMP to measure parallelism efficiency in addition for state-vector quantum simulations. This profiler integrates directly with PennyLane and Qiskit circuits.

## Features

- **Gate timing** with microsecond-precision via RAII (Resource Acquisition Is Initialization) approach using `ScopedTimer` object (wall + CPU time).
- **Memory profiling** - resident set size (RSS) tracked per gate on Linux
- **Gate throughput** in MGates/s (Million Gates per second) with L2/L3 cache boundary detection
- **OpenMP parallelism** - Efficiency denoted by the (CPU time / wall time ratio) (>1 indicates Parallelism effective)
- **Plugin kernel API** - User can register any Python callable or unitary matrix as a profiled gate
- **PennyLane and Qiskit adapters** - Profile circuits from well known SDK circuits without modifying them, can be extended to other SDKs
- **7-panel visual report** (PNG + PDF) with dark-theme matplotlib figures
- **Zero-copy NumPy views** of the C++ state vector via pybind11

## How is it different from other tools?

### Comparison with qprof ([Suau et al., ACM TQC 2022](https://dl.acm.org/doi/10.1145/3529398))

qprof is a gprof-inspired quantum profiler that analyzes quantum circuits by examining their structure — gate counts, circuit depth, T-count (non-Clifford gates relevant to fault tolerance), and qubit width. It wraps circuits from Qiskit and myQLM into a framework-agnostic intermediate representation (qcw), constructs a call graph of subroutine invocations, and exports a report in a gprof-compatible format.

The key distinction is **what each tool actually measures**:

| Dimension | qprof | qprofiler |
|-----------|-------|-----------|
| **Profiling approach** | Static circuit analysis — no execution | Runtime execution — simulation actually runs |
| **Primary question answered** | How algorithmically complex is my circuit? | How fast and memory-efficient is my simulation? |
| **Timing** | Absent — no wall-clock or CPU time | Microsecond wall + CPU time per gate via `ScopedTimer` |
| **Memory** | Tracked in terms of qubit number | RSS measured per gate from `/proc/self/status` |
| **Gate throughput** | Not applicable | MGates/s with L2/L3 cache boundary detection |
| **Parallelism** | Not applicable | CPU/Wall ratio reveals OpenMP thread efficiency |
| **Metrics reported** | Gate count, depth, T-count, circuit width | Wall time, CPU time, RSS, throughput, per-gate records |
| **Call graph** | Yes — subroutine hierarchy (gprof-style) | No — per-gate flat timing table |
| **Dynamic circuits** | Not supported (static analysis only) | Supported — any sequence of gate calls is profiled |
| **Classical host overhead** | Not measured | Python/C++ boundary overhead explicitly measured and plotted |
| **Custom kernels** | Not applicable | Python callables and unitary matrices profiled identically to built-in C++ gates |
| **Framework support** | Qiskit, myQLM (via qcw) | PennyLane, Qiskit (via adapters with unitary fallback) |
| **Output** | gprof-compatible text report, call graph | 7-panel PNG/PDF with dark-theme matplotlib figures |

**In short:** qprof tells you about the quantum algorithm — it operates on the circuit definition and is framework-aware but execution-unaware. qprofiler tells you about the classical simulation — it operates at the C++ gate-kernel level and measures how your hardware actually performs the computation. The two tools answer complementary questions and can be used together: qprof to audit circuit complexity, qprofiler to benchmark simulation efficiency.

## Architecture

```
qprofiler/
├── src/                        # C++ core
│   ├── gates.cpp / .hpp        # Gate kernels (Hadamard, CNOT, Phase, ...) with OpenMP
│   ├── profiler.cpp / .hpp     # ScopedTimer, ProfileRecord, Profiler
│   ├── simulator.cpp / .hpp    # Simulator class, BenchmarkResult
│   ├── kernel_registry.cpp     # KernelRegistry singleton + built-in aliases
│   └── bindings.cpp            # pybind11 module (qprofiler_core)
├── python/
│   ├── qprofiler/              # Python package (imports the .so extension)
│   ├── benchmark.py            # Sweep harness, per-gate profiling, cProfile boundary
│   ├── visualize.py            # 6-panel report generator
│   └── adapters/
│       ├── generic_adapter.py  # register_callable / register_unitary
│       ├── pennylane_adapter.py
│       └── qiskit_adapter.py
├── examples/
│   ├── demo_callable.py        # Python loop + unitary kernels (no extra deps)
│   ├── demo_pennylane.py       # PennyLane circuit profiling
│   └── demo_qiskit.py          # Qiskit circuit profiling
├── tests/
│   ├── test_gates.cpp          # 7 C++ unit tests (gate correctness, RSS, ScopedTimer)
│   ├── test_profiler.py        # Python integration tests (simulator + visualize)
│   └── test_callable_kernel.py # Pure-Python math tests + C++ integration tests
├── generate_full_report.py     # 7-panel report from qp_bench binary output
├── run_profiler.py             # CLI entry point
├── CMakeLists.txt
└── setup.py
```

## Requirements

### C++
- C++17 compiler (GCC >= 9 or Clang >= 10)
- CMake >= 3.16
- pybind11 >= 2.10
- OpenMP (optional, auto-detected)

### Python
- Python >= 3.10
- numpy >= 2.4
- matplotlib >= 3.10
- pytest >= 9.0
- pennylane >= 0.44 _(optional, for PennyLane adapter)_
- qiskit >= 2.3 _(optional, for Qiskit adapter)_

## Installation

### 1. Clone and build

```bash
git clone <repo-url>
cd qprofiler

# CMake build (recommended)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

This produces:
- `build/qp_bench` — standalone C++ benchmark binary
- `build/test_gates` — C++ unit test binary
- `python/qprofiler/qprofiler_core.*.so` — pybind11 Python extension

### 2. Install Python package

```bash
pip install -e .
# or with uv
uv sync
```

### 3. Optional adapters

```bash
pip install pennylane   # for PennyLane adapter
pip install qiskit      # for Qiskit adapter
```

## Quick Start

### Full profiling pipeline

```bash
python3 run_profiler.py                          # default: 4–22 qubits, depth 5
python3 run_profiler.py --q-max 26 --depth 10   # larger sweep
python3 run_profiler.py --cprofile               # include Python/C++ boundary analysis
```

Reports are saved to `reports/` by default.

### 7-panel benchmark report

```bash
python3 generate_full_report.py
# Saves reports/profiler_report_v2.png and .pdf
```

### Direct C++ benchmark

```bash
./build/qp_bench <q_min> <q_max> <depth>
# Example:
./build/qp_bench 4 20 5
```

---

## Python API

### Basic simulation and profiling

```python
import qprofiler_core as qp

sim = qp.Simulator(16)       # 16-qubit state vector, initialized to |0...0>
sim.hadamard(0)
sim.cnot(0, 1)
sim.phase(2, 3.14159 / 4)   # T-like phase rotation

result = sim.run_circuit(depth=5)
print(result["wall_ms"], result["throughput_mgs"])

sim.profiler().print_summary()   # per-gate wall/CPU table
```

### Register a Python callable as a kernel

```python
import math

def ry_kernel(state, n_qubits, args):
    theta = args.params[0]
    target = args.targets[0]
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    mask = 1 << target
    for i in range(len(state)):
        if not (i & mask):
            a, b = state[i], state[i | mask]
            state[i] = c * a - s * b
            state[i | mask] = s * a + c * b

qp.register_kernel("ry", ry_kernel)
sim.apply_gate("ry", [0], [math.pi / 3])
```

### Register a unitary matrix

```python
import math, numpy as np
from python.adapters.generic_adapter import register_unitary

inv2 = 1 / math.sqrt(2)
H = np.array([[inv2, inv2], [inv2, -inv2]], dtype=complex)
register_unitary("my_h", H, n_targets=1)

sim.apply_gate("my_h", [0], [])
```

### Profile a PennyLane circuit

```python
import pennylane as qml
from python.adapters.pennylane_adapter import profile_pennylane_circuit

def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

sim = qp.Simulator(2)
profile_pennylane_circuit(sim, bell_circuit)
sim.profiler().print_summary()
```

### Profile a Qiskit circuit

```python
from qiskit import QuantumCircuit
from python.adapters.qiskit_adapter import profile_qiskit_circuit

qc = QuantumCircuit(2)
qc.h(0)
qc.cx(0, 1)

sim = qp.Simulator(2)
profile_qiskit_circuit(sim, qc)
sim.profiler().print_summary()
```

## Built-in Gate Aliases

| Gate | Aliases |
|------|---------|
| Hadamard | `h`, `hadamard`, `Hadamard` |
| Pauli-X | `x`, `pauli_x`, `PauliX` |
| Pauli-Z | `z`, `pauli_z`, `PauliZ` |
| CNOT | `cx`, `cnot`, `CNOT` |
| Phase(theta) | `phase`, `rz` |
| S gate (pi/2) | `s` |
| T gate (pi/4) | `t` |

---

## Report Panels

`generate_full_report.py` produces a 7-panel figure:

| Panel | Description |
|-------|-------------|
| 1 | Simulation wall time (log scale) with exponential fit |
| 2 | State-vector memory: measured vs theoretical 2^n x 16 B |
| 3 | Gate throughput (MGates/s) — shows L2/L3 cache boundary |
| 4 | CPU/Wall ratio — OpenMP parallelism efficiency |
| 5 | Dual-axis time + memory — where they diverge |
| 6 | Per-gate timing bar chart (wall vs CPU at 16 qubits) |
| 7 | Python callable overhead — C++ vs Python loop vs NumPy unitary |

---

## Testing

```bash
# C++ unit tests
cmake --build build && ./build/test_gates

# Python tests (requires built extension)
pytest tests/ -v

# Pure Python kernel math tests only (no C++ required)
pytest tests/test_callable_kernel.py -k "pure" -v
```

---

## Design Notes

**RAII profiling** — `ScopedTimer` captures wall and CPU timestamps on construction and records the delta on destruction, requiring no explicit flush or teardown.

**Zero-copy state access** — the C++ `StateVec` is passed to Python kernels as a `py::array_t<complex<double>>` backed by the original buffer. Mutations from Python propagate back to the simulator without copying.

**GIL safety** — Python callables registered via `register_kernel` acquire the GIL for the duration of the kernel call. The `clear_python_kernels()` function removes user kernels before Python shutdown to prevent dangling references.

**Adapter fallback** — when the PennyLane or Qiskit adapter encounters an unknown gate, it derives the unitary matrix via the framework's own API and registers it dynamically as a profiled unitary kernel.

## License

Apache 2.0