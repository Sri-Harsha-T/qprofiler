"""
Measures and compares wall time for three kernel types at each qubit count:
  1. Built-in C++ kernel  (apply_hadamard via qp_bench subprocess)
  2. Python loop kernel   (RY rotation — hand-written amplitude loop)
  3. NumPy unitary kernel (Hadamard via 2x2 matrix-vector multiply)

Outputs a CSV + the data needed by the full report generator.

Run:
    python3 examples/overhead_comparison.py
"""

import sys, os, subprocess, re, time, math
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#  Python kernel implementations (mirror what adapters register) 

def _ry_loop_kernel(state: np.ndarray, theta: float, target: int) -> None:
    """Pure Python amplitude-pair loop — same algorithm as C++ hadamard."""
    c, s = math.cos(theta / 2), math.sin(theta / 2)
    mask = 1 << target
    for i in range(len(state)):
        if not (i & mask):
            a, b = state[i], state[i | mask]
            state[i]        =  c * a - s * b
            state[i | mask] =  s * a + c * b


def _h_unitary_kernel(state: np.ndarray, target: int) -> None:
    """NumPy 2x2 matrix-vector multiply — same as register_unitary() path."""
    inv2 = 1.0 / math.sqrt(2.0)
    U = np.array([[inv2, inv2], [inv2, -inv2]])
    mask = 1 << target
    N = len(state)
    for i in range(N):
        if not (i & mask):
            v = np.array([state[i], state[i | mask]])
            w = U @ v
            state[i], state[i | mask] = w[0], w[1]


def _fresh_state(n: int) -> np.ndarray:
    s = np.zeros(2 ** n, dtype=complex)
    s[0] = 1.0
    return s


#  Timer 

def _time_kernel(fn, n_qubits: int, reps: int = 5) -> float:
    """Return median wall time in ms over `reps` runs."""
    times = []
    for _ in range(reps):
        state = _fresh_state(n_qubits)
        t0 = time.perf_counter()
        fn(state)
        times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


#  C++ baseline via qp_bench subprocess 

def _get_cpp_wall_ms(bench_bin: str, q_min: int, q_max: int,
                     depth: int = 1) -> dict[int, float]:
    """
    Run qp_bench and extract per-gate wall times.
    Uses depth=1 and divides by gates-per-layer to get single-gate cost.
    """
    result = subprocess.run(
        [bench_bin, str(q_min), str(q_max), str(depth)],
        capture_output=True, text=True
    )
    rows = {}
    for line in result.stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+\d+\s+\d+\s+([\d.]+)", line)
        if m:
            q    = int(m.group(1))
            wall = float(m.group(2))
            rows[q] = wall   # total circuit wall time; divide later
    return rows


#  Main comparison 

def run_comparison(q_min: int = 4, q_max: int = 20,
                   bench_bin: str = "./qp_bench",
                   reps: int = 5) -> list[dict]:
    """
    Returns a list of dicts, one per qubit count:
        n_qubits, state_mb, cpp_ms, py_loop_ms, numpy_ms,
        loop_ratio, numpy_ratio
    """
    print(f"\nOverhead comparison: {q_min}–{q_max} qubits, {reps} reps each")
    print(f"{'Qubits':>7}  {'StateMB':>8}  {'C++(ms)':>10}  "
          f"{'PyLoop(ms)':>12}  {'NumPy(ms)':>11}  "
          f"{'Loop/C++':>9}  {'NumPy/C++':>10}")
    print("  " + "-" * 76)

    # C++ circuit timing (depth=5) converted to per-gate estimate
    cpp_circuit = _get_cpp_wall_ms(bench_bin, q_min, q_max, depth=5)

    results = []
    for q in range(q_min, q_max + 1):
        state_mb = (2 ** q) * 16 / (1024 ** 2)

        # C++ single-gate estimate: circuit wall / gates per circuit
        gates_per_circuit = 5 * (q + q // 2 + q)   # H + CNOT + Phase layers
        cpp_circuit_ms = cpp_circuit.get(q, float("nan"))
        cpp_ms = cpp_circuit_ms / max(gates_per_circuit, 1)

        # Python loop (RY)
        py_ms = _time_kernel(
            lambda s, _q=q: _ry_loop_kernel(s, math.pi / 3, 0), q, reps)

        # NumPy unitary (H matrix)
        np_ms = _time_kernel(
            lambda s, _q=q: _h_unitary_kernel(s, 0), q, reps)

        loop_ratio  = py_ms  / max(cpp_ms, 1e-9)
        numpy_ratio = np_ms  / max(cpp_ms, 1e-9)

        results.append({
            "n_qubits":    q,
            "state_mb":    state_mb,
            "cpp_ms":      cpp_ms,
            "py_loop_ms":  py_ms,
            "numpy_ms":    np_ms,
            "loop_ratio":  loop_ratio,
            "numpy_ratio": numpy_ratio,
        })

        print(f"  {q:5d}  {state_mb:8.3f}  {cpp_ms:10.4f}  "
              f"{py_ms:12.4f}  {np_ms:11.4f}  "
              f"{loop_ratio:9.1f}x  {numpy_ratio:9.1f}x")

    return results


def save_csv(results: list[dict], path: str) -> None:
    import csv
    keys = list(results[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(results)
    print(f"\nCSV saved → {path}")


if __name__ == "__main__":
    bench = os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "build","qp_bench")
    if not os.path.exists(bench):
        print(f"qp_bench not found at {bench}\n"
              "Build: g++ -std=c++17 -O3 -fopenmp -I src \\\n"
              "       src/gates.cpp src/profiler.cpp src/kernel_registry.cpp \\\n"
              "       src/simulator.cpp src/benchmark_main.cpp -o qp_bench")
        sys.exit(1)

    results = run_comparison(q_min=4, q_max=20, bench_bin=bench)
    os.makedirs("reports", exist_ok=True)
    save_csv(results, "reports/overhead_comparison.csv")
