"""
Profiles a PennyLane QNode or circuit function using qprofiler.

How it works:
1. We use PennyLane's QuantumTape context to record all gate operations without executing
    them on any device.
2. Each recorded operation has .name, .wires, and .parameters.
3. For gates that have a built in qprofiler kernel (H, X, Z, CNOT, ...) we call
    sim.apply_gate(name, targets, params) directly.
4. For unknown gates we query PennyLane for the gates' 2x2 or 4x4 unitary matrix
    and register it on the fly using register_unitary

The profiler handles the full PennyLane gate set includeing Rot, RX, RY, RZ, PhaseShift
and control gates

Usage (Require pennylane installation):

>>> import pennylane as qml
>>> from python.adapters.pennylane_adapter import profile_pennylane_circuit

>>> @qml.qnode(qml.device("default.qubit", wires=4))
... def bell_circuit():
...     qml.Hadamard(wires=0)
...     qml.CNOT(wires=[0, 1])
...     qml.RY(1.23, wires=2)
...     return qml.state()

>>> import qprofiler.qprofiler_core as qc
>>> sim = qc.Simulator(4)
>>> profile_pennylane_circuit(sim, bell_circuit)
>>> sim.print_summary()
"""

from __future__ import annotations

from typing import Callable, List, Tuple
import numpy as np

try:
    import pennylane as qml
    _PL = True
except ImportError:
    _PL = False

try:
    import qprofiler.qprofiler_core as _core
    _CPP = True
except ImportError:
    _CPP = False

# Name mapping: PennyLane gate names to qprofiler kernel names
# Built in kernels in KernelRegistry that have direct equivalents
_PL_TO_QP: dict[str, str] = {
    "Hadamard": "hadamard",
    "PauliX": "pauli_x",
    "PauliY": None, # no built in, will use unitary fallback
    "PauliZ" : "pauli_z",
    "CNOT": "cnot",
    "S": "s",
    "T": "t",
    "PhaseShift": "phase", # params[0] = phi
    "RZ": "rz", # params[0] = theta
}

def _extract_tape_ops(circuit_fn: Callable, *args, **kwargs) -> List[Tuple[str, List[int], List[float]]]:
    """
    Run `circuit_fn` inside a QuantumTape context to record gate operations without
    executing them. Returns a list of (gate_name, targets, params)
    """
    if not _PL:
        raise ImportError("PennyLane is required: pip install pennylane")
    
    with qml.tape.QuantumTape() as tape:
        circuit_fn(*args, **kwargs)

    ops = []
    for op in tape.operations:
        name = op.name
        targets = [w if isinstance(w, int) else int(w) for w in op.wires]
        params = [float(p) for p in op.parameters]
        ops.append((name, targets, params))

    return ops

def _register_pl_gate_if_needed(op_name: str, targets: List[int], params: List[float]) -> str:
    """
    Ensure a kernel exists in qprofiler for this PennyLane gate
    Returns the qprofiler kernel name to use

    For known mappings, return the mapped name directly.
    For unknown gates, computes the unitary matrix via PennyLane 
    and registers it as a new kernel.
    """
    qp_name = _PL_TO_QP.get(op_name)
    if qp_name is not None and _core.has_kernel(qp_name):
        return qp_name
    
    # Fallback: derive the kernel name and register via unitary
    kernel_name = f"pl_{op_name}"
    if params:
        param_str = "_".join(f"{p:.4f}" for p in params) # Include params in name so different angles get distinct entries
        kernel_name = f"pl_{op_name}_{param_str}"
    
    if not _core.has_kernel(kernel_name):
        _register_from_pl_unitary(kernel_name, op_name, targets, params)

    return kernel_name

def _register_from_pl_unitary(kernel_name: str, op_name: str, targets: List[int], params: List[float]) -> None:
    """Compute the gate matrix via PennyLane and register it as a unitary kernel"""
    from python.adapters.generic_adapter import register_unitary

    n_targets = len(targets)
    pl_op = getattr(qml, op_name)(*params, wires=list(range(n_targets))) # Build PennyLane operation to get its matrix
    matrix = qml.matrix(pl_op)

    register_unitary(kernel_name, matrix, n_targets=n_targets)

# Public API

def profile_pennylane_circuit(sim, circuit_fn: Callable, *args, reset: bool = True, **kwargs) -> None:
    """
    Profile every gate in a PennyLane circuit function using qprofiler

    Intercepts the PennyLane tape, maps each gate to a qprofiler kernel (registering unitary based
    fallback kernels on the fly), and applies them through sim.apply_gate() so every gate 
    is wrapped in ScopedTimer

    Parameters:
    sim: qprofiler.qprofiler_core.Simulator
    circuit_fn: callable, a plain Python function (or unwrapped QNode body)
                that calls PennyLane gate operations
    *args: forwarded to circuit_fn
    reset: if True, reset the state vector to |0...0> before running
    **kwargs: forwarded to circuit_fn

    Example
    -------
    >>> def my_circuit():
    ...     qml.Hadamard(wires=0)
    ...     qml.CNOT(wires=[0, 1])
    ...     qml.RY(1.5, wires=2)

    >>> profile_pennylane_circuit(sim, my_circuit)
    >>> sim.print_summary()
    """

    if not _CPP:
        raise RuntimeError("qprofiler.qprofiler_core C++ extension not available")
    if not _PL:
        raise ImportError("PennyLane is required: pip install pennylane")
    
    ops = _extract_tape_ops(circuit_fn, *args, **kwargs)
    if reset:
        sim.reset_state()
    for (op_name, targets, params) in ops:
        kernel_name = _register_pl_gate_if_needed(op_name, targets, params)
        sim.apply_gate(kernel_name, targets, params)

def extract_op_list(circuit_fn: Callable, *args, **kwargs) -> List[Tuple[str, List[int], List[float]]]:
    """
    Return the raw (gate_name, targets, params) list from a PennyLane circuit without running it
    through the profiler. Useful for inspection.
    """
    return _extract_tape_ops(circuit_fn, *args, **kwargs)

