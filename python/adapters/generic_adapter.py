"""
Adapter that lets any Python function or a raw NumPy unitary matrix
be profiled as a named gate kernel in qprofiler.

Works standalon (No PennyLane/Qiskit required)

Examples:
>>> from python.adapters.generic_adapter import register_callable, register_unitary
>>> import numpy as np
>>> import qprofiler.qprofiler_core as qc

>>> # 1. Register a hand-written kernel function
>>> def ry_kernel(state, n_qubits, targets, params):
...     theta = params[0]
...     c, s  = np.cos(theta / 2), np.sin(theta / 2)
...     mask  = 1 << targets[0]
...     for i in range(len(state)):
...         if not (i & mask):
...             a, b = state[i], state[i | mask]
...             state[i]        =  c * a - s * b
...             state[i | mask] =  s * a + c * b

>>> register_callable("RY", ry_kernel)

>>> # 2. Register via a 2x2 NumPy unitary matrix
>>> T_matrix = np.array([[1, 0], [0, np.exp(1j * np.pi / 4)]], dtype=complex)
>>> register_unitary("T_custom", T_matrix)

>>> # 3. Profile a circuit using both
>>> sim = qc.Simulator(14)
>>> sim.apply_gate("RY",       targets=[0], params=[np.pi / 3])
>>> sim.apply_gate("T_custom", targets=[1])
>>> sim.print_summary()
"""

from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional

try:
    import qprofiler.qprofiler_core as _core
    _CPP = True
except ImportError:
    _CPP = False

# Type alias
KernelCallable = Callable[[np.ndarray, int, List[int], List[float]], None]

# Registration helpers
def register_callable(name: str, fn: KernelCallable)-> None:
    """
    Register any Python function as a named gate kernel.
    The function must mutate the state array in-place:
        def fn(state: np.ndarray, # complex128, shape(2^n)
               n_qubits: int,
               targets: list[int],
               params: list[float]) -> None

    Parameters:
    name: str, kernel name used in apply_gate() and ProfileRecord labels
    fn: callable, satisfies the signature above
    """

    if not _CPP:
        raise RuntimeError("qprofiler_core C++ extension not available")
    _core.register_kernel(name, fn)

def register_unitary(name: str, matrix: np.ndarray, n_targets: int = 1)->None:
    """
    Register a gate defined by its unitary matrix.
    Builds a kernel function that applies the matrix to the specified target
    qubits using dense matrix-vector multiplication. Supports 1-qubit(2x2) and
    2-qubit (4x4) unitaries.

    Parameters:
    name: str , # kernel name
    matrix: np.ndarray, # unitary matrix, shape (2^k, 2^k), complex128
    n_targets: int # number of target qubits (1 or 2)
    """

    if not _CPP:
        raise RuntimeError("qprofiler_core C++ extension not available")
    matrix = np.asarray(matrix, dtype=complex)
    expected = (2**n_targets, 2**n_targets)
    if matrix.shape != expected:
        raise ValueError(
            f"register_unitary '{name}': expected shape {expected}, "
            f"got {matrix.shape}"
        )
    
    if n_targets == 1:
        _core.register_kernel(name, _make_1q_unitary_kernel(matrix))
    elif n_targets == 2:
        _core.register_kernel(name, _make_2q_unitary_kernel(matrix))
    else:
        raise ValueError("register_unitary: n_targets must be 1 or 2")
    
# Unitary kernel builders

def _make_1q_unitary_kernel(U: np.ndarray) -> KernelCallable:
    """
    Return a kernel function that applies a 2x2 unitary U to one qubit.

    Uses the same amplitude-pair loop pattern as the C++ built-in gates.
    For the |0⟩ / |1⟩ amplitudes of qubit k:
        [new_a]   [U00  U01] [a]
        [new_b] = [U10  U11] [b]
    """
    u00, u01, u10, u11 = U[0, 0], U[0, 1], U[1, 0], U[1, 1]

    def kernel(state: np.ndarray, n_qubits: int,
               targets: List[int], params: List[float]) -> None:
        if not targets:
            raise ValueError("1-qubit unitary kernel: no target qubit")
        mask = 1 << targets[0]
        N    = len(state)
        for i in range(N):
            if not (i & mask):
                a = state[i]
                b = state[i | mask]
                state[i]        = u00 * a + u01 * b
                state[i | mask] = u10 * a + u11 * b

    return kernel


def _make_2q_unitary_kernel(U: np.ndarray) -> KernelCallable:
    """
    Return a kernel function that applies a 4x4 unitary U to two qubits.

    Basis ordering: |ctrl_bit  tgt_bit⟩ — low bit = target, high bit = ctrl.
    The four amplitude indices for a given (i, ctrl, tgt) are:
        i00 = i (ctrl=0, tgt=0)
        i01 = i | tgt_mask
        i10 = i | ctrl_mask
        i11 = i | ctrl_mask | tgt_mask
    """
    def kernel(state: np.ndarray, n_qubits: int,
               targets: List[int], params: List[float]) -> None:
        if len(targets) < 2:
            raise ValueError("2-qubit unitary kernel: need 2 targets")
        ctrl_mask = 1 << targets[0]
        tgt_mask  = 1 << targets[1]
        N = len(state)

        for i in range(N):
            # Only process the "00" element of each 4-element group
            if (i & ctrl_mask) or (i & tgt_mask):
                continue
            i00, i01 = i, i | tgt_mask
            i10, i11 = i | ctrl_mask, i | ctrl_mask | tgt_mask

            v = np.array([state[i00], state[i01],
                          state[i10], state[i11]])
            w = U @ v
            state[i00], state[i01] = w[0], w[1]
            state[i10], state[i11] = w[2], w[3]

    return kernel

# Circuit operation list runner

def run_op_list(sim, ops: List[tuple], fallback: Optional[Callable] = None) -> None:
    """
    Apply a list of (gate_name, targets, params) tuples to the simulator.
    Each operation is applied via sim.apply_gate(), which wraps it in a ScopedTimer.
    Unknown kernels are passed to `fallback` if provided, otherwise a KeyError is raised.

    Parameters:

    sim: qprofiler_core.Simulator
    ops: list of (name: str, targets: list[int], params: list[float])
    fallback: optional callable(sim, name, targets, params) for unknown gates
    """
    for name, targets, params in ops:
        if _core.has_kernel(name):
            sim.apply_gate(name, targets, params)
        elif fallback is not None:
            fallback(sim, name, targets, params)
        else:
            raise KeyError(
                f"run_op_list: gate '{name}' not registered. "
                f"Call register_callable('{name}', fn) first, "
                f"or provide a fallback."
            )

    