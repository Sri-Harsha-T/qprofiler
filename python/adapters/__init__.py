"""
python/adapters — plug-in kernel adapters for qprofiler.

Exports
-------
register_callable     register any Python function as a named kernel
register_unitary      register a gate via its 2x2 or 4x4 unitary matrix
run_op_list           apply a (name, targets, params) list to a Simulator

profile_pennylane_circuit   profile a PennyLane circuit function
extract_op_list             inspect PennyLane tape ops without profiling

profile_qiskit_circuit      profile a Qiskit QuantumCircuit
"""

from .generic_adapter import register_callable, register_unitary, run_op_list

# Soft imports — only fail at call-time if the library is absent
try:
    from .pennylane_adapter import profile_pennylane_circuit, extract_op_list as pl_op_list
except ImportError:
    pass

try:
    from .qiskit_adapter import profile_qiskit_circuit, extract_op_list as qk_op_list
except ImportError:
    pass

__all__ = [
    "register_callable",
    "register_unitary",
    "run_op_list",
    "profile_pennylane_circuit",
    "profile_qiskit_circuit",
]
