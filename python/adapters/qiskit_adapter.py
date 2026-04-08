"""
Profiles a Qiskit QuantumCircuit using qprofiler.

How it works
1. `circuit.data` is a list of CircuitInstruction objects, each holding an `operation`
    (the gate), a list of `qubits` and `clbits`.
2. For gates with a built-in qprofiler kernel (h, x, z, cx) we call sim.apply_gate()
   directly with the mapped name.
3. For any other gate we call `gate.to_matrix()` to get the unitary and
   register it on the fly.

Usage:

>>> from qiskit import QuantumCircuit
>>> from python.adapters.qiskit_adapter import profile_qiskit_circuit

>>> qc = QuantumCircuit(4)
>>> qc.h(0)
>>> qc.cx(0, 1)
>>> qc.ry(1.23, 2)
>>> qc.t(3)

>>> import qprofiler.qprofiler_core as qp
>>> sim = qp.Simulator(4)
>>> profile_qiskit_circuit(sim, qc)
>>> sim.print_summary()
"""

from __future__ import annotations

from typing import List, Tuple
import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Instruction
    _QISKIT = True
except ImportError:
    _QISKIT = False

try:
    import qprofiler.qprofiler_core as _core
    _CPP = True
except ImportError:
    _CPP = False

# Name mapping: Qiskit gate name to qprofiler kernel name
_QISKIT_TO_QP: dict[str, str] = {
    "h": "hadamard",
    "x": "pauli_x",
    "z": "pauli_z",
    "cx": "cnot",
    "s": "s",
    "t": "t",
    "rz": "rz",
    "p": "phase",    # Qiskit Phase gate = diag(1, e^{i*theta})
}

def _qubit_index(qubit, circuit) -> int:
    """Convert a Qiskit Qubit object to a plain integer index"""
    return circuit.find_bit(qubit).index

def _qiskit_op_list(circuit) -> List[Tuple[str, List[int], List[float]]]:
    """Extract (gate_name, targets, params) from a Qiskit QuantumCircuit"""
    if not _QISKIT:
        raise ImportError("Qiskit is required: pip install qiskit")

    ops = []
    for instr in circuit.data:
        op = instr.operation
        qubits = [_qubit_index(q, circuit) for q in instr.qubits]
        params = [float(p) for p in op.params]
        ops.append((op.name, qubits, params))
    return ops

def _register_qiskit_gate_if_needed(op_name: str, op, targets: List[int], params: List[float]) -> str:
    """
    Ensure a qprofiler kernel exists for this Qiskit gate.
    Returns the kernel name to pass to apply_gate().
    """
    qp_name = _QISKIT_TO_QP.get(op_name)
    if qp_name and _core.has_kernel(qp_name):
        return qp_name

    # Build a unique kernel name
    kernel_name = f"qk_{op_name}"
    if params:
        param_str = "_".join(f"{p:.4f}" for p in params)
        kernel_name = f"qk_{op_name}_{param_str}"

    if not _core.has_kernel(kernel_name):
        _register_from_qiskit_gate(kernel_name, op, targets)

    return kernel_name

def _register_from_qiskit_gate(kernel_name: str, op, targets: List[int]) -> None:
    """Compute the gate matrix via Qiskit and register as a unitary kernel"""
    from python.adapters.generic_adapter import register_unitary
    try:
        matrix = op.to_matrix()
    except Exception as exc:
        raise RuntimeError(
            f"Cannot compute unitary matrix for Qiskit gate '{op.name}': {exc}\n"
            f"Register it manually with register_callable('{kernel_name}', fn)."
        ) from exc

    n_targets = len(targets)
    register_unitary(kernel_name, matrix, n_targets=n_targets)

# Public API

def profile_qiskit_circuit(sim, circuit, reset: bool = True) -> None:
    """
    Profile every gate in a Qiskit QuantumCircuit using qprofiler.
    Walks `circuit.data`, maps each gate to a qprofiler kernel (registering unitary
    based fallbacks on the fly), and applies them via sim.apply_gate() so every
    gate is wrapped in ScopedTimer.

    Parameters
    sim: qprofiler.qprofiler_core.Simulator
    circuit: qiskit.QuantumCircuit
    reset: if True, reset the state vector to |0...0> before running

    Example

    >>> from qiskit import QuantumCircuit
    >>> qc = QuantumCircuit(4)
    >>> qc.h(0); qc.cx(0, 1); qc.ry(1.5, 2)

    >>> sim = qp.Simulator(4)
    >>> profile_qiskit_circuit(sim, qc)
    >>> sim.print_summary()
    """
    if not _CPP:
        raise RuntimeError("qprofiler.qprofiler_core C++ extension not available.")
    if not _QISKIT:
        raise ImportError("Qiskit is required: pip install qiskit")

    if reset:
        sim.reset_state()

    for instr in circuit.data:
        op = instr.operation
        qubits = [_qubit_index(q, circuit) for q in instr.qubits]
        params = [float(p) for p in op.params]

        kernel_name = _register_qiskit_gate_if_needed(op.name, op, qubits, params)
        sim.apply_gate(kernel_name, qubits, params)

def extract_op_list(circuit) -> List[Tuple[str, List[int], List[float]]]:
    """Return the (gate_name, targets, params) list without profiling"""
    return _qiskit_op_list(circuit)
