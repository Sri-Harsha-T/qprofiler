"""
Demo of the qprofiler (Quantum Performance Profiler)

Usage :
python run_profiler.py # default sweep (4-22) qubits
python run_profiler.py --q-max 26 # extend sweep
python run_profiler.py --no-report # skip visual report
python run_profiler.py --cprofile # print cProfile boundary analysis
"""

import sys
import os
import argparse

# Setup path and arguments
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

def parse_args():
    p = argparse.ArgumentParser(description="Quantum Profiler demo runner")
    p.add_argument("--q-min", type=int, default=4, help="Min qubits (default: 4)")
    p.add_argument("--q-max", type=int, default=22, help="Max qubits (default: 22)")
    p.add_argument("--depth", type=int, default=5, help="Circuit depth (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument("--no-report", action="store_true", help="Skip visual report")
    p.add_argument("--cprofile", action="store_true", help="Print cProfile analysis")
    p.add_argument("--output-dir", default="reports", help="Report output directory")
    return p.parse_args()

def main():
    args = parse_args()

    # Import C++ Pybind generated library
    try:
        import qprofiler.qprofiler_core as _core
        print(f"C++ extension loaded (OPENMP: {_core.openmp_enabled})")
        print(f"Curent RSS: {_core.current_rss_kb()} kB\n")
    except ImportError:
        print("C++ extension not found")
        print("Build instructions:")
        print("cmake -B build -DCMAKE_BUILD_TYPE=Release")
        print("cmake --build build -j$(nproc)")
        sys.exit(1)

    from benchmark import run_sweep, profile_gate_sequence, cprofile_boundary
    from visualize import generate_report

    # Main qubit sweep
    results = run_sweep(q_min = args.q_min, q_max=args.q_max, depth=args.depth, seed=args.seed, verbose=True)

    # Gate level profiling
    q_gate = min(16, args.q_max)
    print(f"Gate level profiling at {q_gate} qubits ...")
    gate_records = profile_gate_sequence(n_qubits=q_gate)
    print(f"\n  {'Gate':<12} {'Wall(ms)':>10} {'CPU(ms)':>10} {'RSS(kB)':>10}")
    print("  " + "-" * 46)
    for r in gate_records:
        print(f"  {r.label:<12} {r.wall_ms:>10.4f} {r.cpu_ms:>10.4f} {r.peak_rss_kb:>10}")

    # Optional cProfile boundary analysis
    if args.cprofile:
        print("\ncProfile: Python - C++ boundary")
        q_prof = min(18, args.q_max)
        report_str = cprofile_boundary(n_qubits=q_prof, depth=3, top_n=15)
        print(report_str)

    # Performance report
    if not args.no_report:
        try:
            png = generate_report(sweep_results = results, gate_records = gate_records, output_dir = args.output_dir)
            print(f"Report : {png}")
        except ImportError as e:
            print(f"Report skipped: {e}")

    # Summary statistics
    print("Scaling summary")
    if len(results) >= 2:
        import numpy as np
        qubits = np.array([r.n_qubits for r in results])
        walls = np.array([r.wall_ms for r in results])
        valid = walls > 0
        k = np.polyfit(qubits[valid], np.log2(walls[valid] + 1e-12), 1)[0]
        print(f"  Wall time scaling: ~2^({k:.3f} x n_qubits)")
        print(f"  Peak RSS at {results[-1].n_qubits}q: {results[-1].peak_rss_kb} kB")
        print(f"  State-vector at {results[-1].n_qubits}q: {results[-1].state_mb:.1f} MB"
              f"  (theoretical: {(2**results[-1].n_qubits)*16/(1024**2):.1f} MB)")
        max_thr = max(r.throughput_mgs for r in results)
        print(f"  Peak throughput: {max_thr:.2f} MGates/s")

    print("Done.")

if __name__ == "__main__":
    main()
    