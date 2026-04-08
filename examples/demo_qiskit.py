"""
Profile Qiskit QuantumCircuit objects using qprofiler

Requirements : Qiskit | pip install qiskit

Run : python3 examples/demo_qiskit.py
"""

import sys, os
# sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import QFT
except ImportError:
    print("Qiskit not installed.  Run: pip install qiskit")
    sys.exit(0)

try:
    import qprofiler.qprofiler_core as qc
except ImportError:
    print("qprofiler_core not built.  Run: cmake -B build && cmake --build build")
    sys.exit(1)

from python.adapters.qiskit_adapter import (
    profile_qiskit_circuit, extract_op_list
)

N = 6

#  Circuit 1: Bell state 

qc_bell = QuantumCircuit(N)
qc_bell.h(0)
qc_bell.cx(0, 1)

print("=" * 56)
print("  Qiskit → qprofiler  (Bell state)")
print("=" * 56)

print("\nGates extracted from circuit.data:")
for name, targets, params in extract_op_list(qc_bell):
    print(f"  {name:12s}  qubits={targets}  params={params}")

sim = qc.Simulator(N)
profile_qiskit_circuit(sim, qc_bell)
print("\nProfiler summary:")
sim.print_summary()

#  Circuit 2: Parametric circuit (RY, RZ — uses built-in rz kernel) 

qc_param = QuantumCircuit(N)
for i in range(N):
    qc_param.ry(np.pi / (i + 1), i)
for i in range(N - 1):
    qc_param.cx(i, i + 1)
for i in range(N):
    qc_param.rz(np.pi / (N - i), i)

print("\n" + "=" * 56)
print("  Parametric circuit (RY via unitary, RZ built-in)")
print("=" * 56)

sim2 = qc.Simulator(N)
profile_qiskit_circuit(sim2, qc_param)
sim2.print_summary()

#  Circuit 3: QFT — showcases unitary fallback for swap + cp gates 

try:
    qft = QFT(N, do_swaps=True).decompose()

    print("\n" + "=" * 56)
    print(f"  QFT on {N} qubits (full gate set with unitary fallback)")
    print("=" * 56)
    print(f"  Gate count in circuit: {len(qft.data)}")

    sim3 = qc.Simulator(N)
    profile_qiskit_circuit(sim3, qft)
    records3 = sim3.get_records()

    # Summarise by gate type
    from collections import Counter
    label_counts = Counter(r["label"] for r in records3)
    print(f"\n  Gate profile ({len(records3)} total gates):")
    for label, count in sorted(label_counts.items()):
        avg_ms = np.mean([r["wall_ms"] for r in records3 if r["label"] == label])
        print(f"    {label:20s}  {count:3d}×   avg {avg_ms:.4f} ms")

except Exception as exc:
    print(f"  QFT demo skipped: {exc}")

#  Overhead comparison 
records2 = sim2.get_records()
builtin_labels = {"hadamard", "cnot", "pauli_x", "pauli_z", "phase", "rz", "cx", "h", "x", "z"}
builtin_ms = [r["wall_ms"] for r in records2 if r["label"] in builtin_labels]
unitary_ms = [r["wall_ms"] for r in records2 if r["label"].startswith("qk_")]

if builtin_ms:
    print(f"\nOverhead summary (parametric circuit):")
    print(f"  Built-in gates avg: {np.mean(builtin_ms):.4f} ms")
if unitary_ms:
    print(f"  Unitary gates avg:  {np.mean(unitary_ms):.4f} ms")
    print(f"  Overhead ratio:     {np.mean(unitary_ms)/max(np.mean(builtin_ms),1e-9):.1f}×")

print("\nFinal registry:")
for k in qc.list_kernels():
    print(f"  {k}")
