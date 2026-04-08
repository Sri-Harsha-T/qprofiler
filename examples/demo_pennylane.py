"""
Profile PennyLane circuits using qprofiler.

Requirements : PennyLane | pip install pennylane

Run : python3 examples/demo_pennylane.py
"""

import sys, os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import pennylane as qml
except ImportError:
    print("PennyLane not installed.  Run: pip install pennylane")
    sys.exit(0)

try:
    import qprofiler.qprofiler_core as qc
except ImportError:
    print("qprofiler.qprofiler_core not built.  Run: cmake -B build && cmake --build build")
    sys.exit(1)

from python.adapters.pennylane_adapter import (
    profile_pennylane_circuit, extract_op_list
)

N = 6   # qubits

#  Circuit 1: Bell state preparation 

def bell_circuit():
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])

print("=" * 56)
print("  PennyLane → qprofiler  (Bell state, 6q simulator)")
print("=" * 56)

print("\nTape ops extracted (no device execution):")
for name, targets, params in extract_op_list(bell_circuit):
    print(f"  {name:16s}  wires={targets}  params={params}")

sim = qc.Simulator(N)
profile_pennylane_circuit(sim, bell_circuit)
print("\nProfiler summary:")
sim.print_summary()

#  Circuit 2: Parametric rotation circuit 

def rotation_circuit(thetas):
    for i, theta in enumerate(thetas):
        qml.RY(theta, wires=i)
    for i in range(len(thetas) - 1):
        qml.CNOT(wires=[i, i + 1])
    for i, theta in enumerate(thetas):
        qml.RZ(theta / 2, wires=i)

thetas = [np.pi / k for k in range(1, N + 1)]

print("\n" + "=" * 56)
print("  Parametric rotation circuit")
print("=" * 56)

sim2 = qc.Simulator(N)
profile_pennylane_circuit(sim2, rotation_circuit, thetas)
sim2.print_summary()

#  Overhead breakdown 
records = sim2.get_records()
builtin_ms = [r["wall_ms"] for r in records if r["label"] in ("hadamard","cnot","pauli_z","pauli_x","phase","rz")]
python_ms  = [r["wall_ms"] for r in records if r["label"].startswith("pl_")]

if builtin_ms and python_ms:
    print(f"\nOverhead summary:")
    print(f"  Built-in gates avg wall_ms : {np.mean(builtin_ms):.4f}")
    print(f"  PL unitary gates avg wall_ms: {np.mean(python_ms):.4f}")
    ratio = np.mean(python_ms) / max(np.mean(builtin_ms), 1e-9)
    print(f"  Overhead ratio             : {ratio:.1f}×")

#  Circuit 3: Grover oracle (shows unitary fallback) 

def grover_step():
    for i in range(N):
        qml.Hadamard(wires=i)
    # CZ — will trigger unitary matrix fallback
    qml.CZ(wires=[0, 1])
    for i in range(N):
        qml.PauliZ(wires=i)

print("\n" + "=" * 56)
print("  Grover step (CZ uses unitary fallback)")
print("=" * 56)

sim3 = qc.Simulator(N)
profile_pennylane_circuit(sim3, grover_step)
sim3.print_summary()

print("\nRegistered kernels after all circuits:")
for k in qc.list_kernels():
    print(f"  {k}")
