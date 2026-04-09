"""
Generates the complete 7-panel performance report from real benchmark data.

Panels
------
1. Simulation wall time (log scale) — exponential scaling with qubit count
2. State-vector memory — measured vs theoretical 2^n x 16 B
3. Gate throughput (MGates/s) — shows L2/L3 cache boundary drop
4. CPU / Wall ratio — OpenMP parallelism efficiency
5. Dual-axis time + memory — where they diverge
6. Per-gate timing (built-in gates at 16 qubits)
7. Python callable overhead — C++ vs Python loop vs NumPy unitary

Run:
    python3 generate_full_report.py
"""

import subprocess, re, sys, os, time, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).parent

#  Palette 
_BG       = "#0D0D1A"
_PANEL    = "#13132A"
_FG       = "#E2E8F0"
_GRID     = "#1E2040"
_C_WALL   = "#A78BFA"
_C_MEM    = "#E879F9"
_C_THRU   = "#34D399"
_C_RATIO  = "#FBBF24"
_C_RSS    = "#38BDF8"
_C_THEORY = "#F87171"
_C_CPP    = "#34D399"
_C_PYLOOP = "#FBBF24"
_C_NUMPY  = "#F87171"


def _style(ax):
    ax.set_facecolor(_PANEL)
    ax.tick_params(colors=_FG, which="both", labelsize=8.5)
    ax.xaxis.label.set_color(_FG); ax.xaxis.label.set_fontsize(9.5)
    ax.yaxis.label.set_color(_FG); ax.yaxis.label.set_fontsize(9.5)
    ax.title.set_color(_FG);       ax.title.set_fontsize(10.5); ax.title.set_fontweight("bold")
    for sp in ax.spines.values(): sp.set_edgecolor(_GRID)
    ax.grid(True, color=_GRID, lw=0.6, ls="--", alpha=0.75)


#  1. Collect C++ benchmark data 

def get_benchmark_data(bench: str, q_min=4, q_max=22, depth=5):
    r = subprocess.run([bench, str(q_min), str(q_max), str(depth)],
                       capture_output=True, text=True)
    qubits, wall, cpu, rss, state_bytes, gates = [], [], [], [], [], []
    for line in r.stdout.splitlines():
        m = re.match(r"^\s*(\d+)\s+\d+\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s+(\d+)", line)
        if m:
            qubits.append(int(m.group(1)))
            state_bytes.append(int(m.group(2)))
            wall.append(float(m.group(3)))
            cpu.append(float(m.group(4)))
            rss.append(int(m.group(5)))
            gates.append(int(m.group(6)))
    return (np.array(qubits), np.array(wall), np.array(cpu),
            np.array(rss), np.array(state_bytes), np.array(gates))


#  2. Collect Python callable overhead data 

def _ry_loop(state, theta, target):
    c, s = math.cos(theta/2), math.sin(theta/2)
    mask = 1 << target
    for i in range(len(state)):
        if not (i & mask):
            a, b = state[i], state[i|mask]
            state[i] = c*a - s*b;  state[i|mask] = s*a + c*b

def _numpy_h(state, target):
    inv2 = 1/math.sqrt(2)
    U = np.array([[inv2,inv2],[inv2,-inv2]])
    mask = 1 << target
    for i in range(len(state)):
        if not (i & mask):
            v = np.array([state[i], state[i|mask]])
            w = U @ v
            state[i], state[i|mask] = w[0], w[1]

def _median_ms(fn, q, reps=5):
    times = []
    for _ in range(reps):
        s = np.zeros(2**q, dtype=complex); s[0] = 1
        t0 = time.perf_counter()
        fn(s)
        times.append((time.perf_counter()-t0)*1000)
    return float(np.median(times))

def get_overhead_data(qubits, cpp_wall, gates_per_circuit):
    cpp_per_gate, py_loop, np_unitary = [], [], []
    print("  Timing Python kernels", end="", flush=True)
    for i, q in enumerate(qubits):
        gpc = max(int(gates_per_circuit[i]), 1)
        cpp_per_gate.append(cpp_wall[i] / gpc)
        py_loop.append(_median_ms(lambda s, _q=q: _ry_loop(s, math.pi/3, 0), q))
        np_unitary.append(_median_ms(lambda s, _q=q: _numpy_h(s, 0), q))
        print(".", end="", flush=True)
    print(" done")
    return (np.array(cpp_per_gate), np.array(py_loop), np.array(np_unitary))


#  3. Per-gate timing (from qp_bench output) 

def get_per_gate_data(bench: str):
    r = subprocess.run([bench, "16", "16", "5"], capture_output=True, text=True)
    labels, wall_ms, cpu_ms = [], [], []
    in_gate_section = False
    for line in r.stdout.splitlines():
        if "Per gate timing" in line:
            in_gate_section = True; continue
        if in_gate_section:
            m = re.match(r"^\s*(\w+)\s+\d+\s+([\d.]+)\s+([\d.]+)", line)
            if m:
                labels.append(m.group(1))
                wall_ms.append(float(m.group(2)))
                cpu_ms.append(float(m.group(3)))
    return labels, np.array(wall_ms), np.array(cpu_ms)


#  4. Build the 7-panel figure 

def build_report(bench: str, out_dir: Path):
    print("Collecting benchmark data...")
    q, wall, cpu, rss, sbytes, gates = get_benchmark_data(bench)

    theory_mb  = (2.0**q) * 16 / (1024**2)
    state_mb   = sbytes  / (1024**2)
    gpc        = 5 * (q + q//2 + q)           # gates per circuit
    throughput = gates / (wall * 1e-3 + 1e-12) / 1e6
    ratio      = np.where(wall > 0, cpu / (wall + 1e-9), 1.0)

    print("Collecting overhead data (timing Python kernels)...")
    cpp_pg, py_ms, np_ms = get_overhead_data(q, wall, gpc)
    loop_ratio  = py_ms  / (cpp_pg + 1e-9)
    numpy_ratio = np_ms  / (cpp_pg + 1e-9)

    gate_labels, g_wall, g_cpu = get_per_gate_data(bench)

    #  Layout 
    fig = plt.figure(figsize=(21, 16), facecolor=_BG)
    fig.suptitle(
        "⚛   Quantum Profiler  ·  Complete Performance Report  "
        "·  C++ / OpenMP / Python Callable",
        fontsize=16, color=_FG, fontweight="bold", y=0.975
    )
    gs = gridspec.GridSpec(2, 4, figure=fig,
                           hspace=0.44, wspace=0.34,
                           left=0.06, right=0.97,
                           top=0.92,  bottom=0.06)
    # 7 panels: 4 top + 3 bottom (last bottom spans 2 cols)
    ax = [
        fig.add_subplot(gs[0, 0]),   # 0: wall time
        fig.add_subplot(gs[0, 1]),   # 1: memory
        fig.add_subplot(gs[0, 2]),   # 2: throughput
        fig.add_subplot(gs[0, 3]),   # 3: cpu/wall ratio
        fig.add_subplot(gs[1, 0]),   # 4: dual-axis
        fig.add_subplot(gs[1, 1]),   # 5: per-gate bar
        fig.add_subplot(gs[1, 2:]), # 6: Python overhead (wide)
    ]
    for a in ax: _style(a)

    #  Panel 0: Wall time 
    a = ax[0]
    a.semilogy(q, wall, "o-", color=_C_WALL, lw=2, ms=6,
               markerfacecolor=_BG, markeredgewidth=1.8, label="Wall time")
    k, b = np.polyfit(q, np.log2(wall + 1e-9), 1)
    fit  = 2**(k*q + b)
    a.semilogy(q, fit, "--", color=_C_THEORY, lw=1.4, alpha=0.7,
               label=f"fit: 2^({k:.2f}·n)")
    a.set_xlabel("Qubits"); a.set_ylabel("Wall time (ms)")
    a.set_title("① Simulation wall time")
    a.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8)
    a.annotate(f"{wall[-1]/1000:.0f}s at {q[-1]}q",
               xy=(q[-1], wall[-1]), xytext=(-55, -20),
               textcoords="offset points", color=_FG, fontsize=7.5,
               arrowprops=dict(arrowstyle="->", color=_FG, lw=0.7))

    #  Panel 1: Memory 
    a = ax[1]
    a.semilogy(q, state_mb,  "s-", color=_C_MEM,    lw=2, ms=6,
               markerfacecolor=_BG, markeredgewidth=1.8, label="Measured")
    a.semilogy(q, theory_mb, "--", color=_C_THEORY, lw=1.4, alpha=0.7,
               label="2ⁿ × 16 B")
    a.set_xlabel("Qubits"); a.set_ylabel("State vector (MB)")
    a.set_title("② State-vector memory")
    a.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8)
    a.annotate(f"{state_mb[-1]:.0f} MB at {q[-1]}q",
               xy=(q[-1], state_mb[-1]), xytext=(-70, 10),
               textcoords="offset points", color=_FG, fontsize=7.5,
               arrowprops=dict(arrowstyle="->", color=_FG, lw=0.7))

    #  Panel 2: Throughput 
    a = ax[2]
    a.plot(q, throughput, "^-", color=_C_THRU, lw=2, ms=6,
           markerfacecolor=_BG, markeredgewidth=1.8)
    a.fill_between(q, throughput, alpha=0.12, color=_C_THRU)
    peak_q = q[np.argmax(throughput)]
    a.axvline(peak_q, color=_C_THRU, lw=1, ls=":", alpha=0.6)
    a.annotate(f"Cache boundary\n~{peak_q}q",
               xy=(peak_q, throughput.max()*0.8), xytext=(5, 0),
               textcoords="offset points", color=_C_THRU, fontsize=7.5)
    a.set_xlabel("Qubits"); a.set_ylabel("Throughput (MGates/s)")
    a.set_title("③ Gate throughput")

    #  Panel 3: CPU/Wall ratio 
    a = ax[3]
    a.plot(q, ratio, "D-", color=_C_RATIO, lw=2, ms=5,
           markerfacecolor=_BG, markeredgewidth=1.8)
    a.axhline(1.0, color=_FG,    lw=0.8, ls=":", alpha=0.4, label="1× (serial)")
    a.axhline(4.0, color=_C_THRU, lw=0.8, ls=":", alpha=0.5, label="4× (4 threads)")
    a.set_xlabel("Qubits"); a.set_ylabel("CPU time / Wall time")
    a.set_title("④ OpenMP efficiency")
    a.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8)

    #  Panel 4: Dual-axis 
    a4a = ax[4]
    a4b = a4a.twinx()
    _style(a4a); a4b.set_facecolor(_PANEL)
    for sp in a4b.spines.values(): sp.set_edgecolor(_GRID)
    l1, = a4a.semilogy(q, wall,     "o-", color=_C_WALL, lw=2, ms=5,
                        markerfacecolor=_BG, markeredgewidth=1.8, label="Wall (ms)")
    l2, = a4b.semilogy(q, state_mb, "s--", color=_C_MEM, lw=2, ms=5,
                        markerfacecolor=_BG, markeredgewidth=1.8, label="State (MB)")
    a4a.set_xlabel("Qubits")
    a4a.set_ylabel("Wall time (ms)", color=_C_WALL)
    a4b.set_ylabel("State vector (MB)", color=_C_MEM)
    a4a.tick_params(axis="y", colors=_C_WALL)
    a4b.tick_params(axis="y", colors=_C_MEM)
    a4a.set_title("⑤ Time vs memory")
    a4a.legend(handles=[l1, l2], facecolor=_GRID, labelcolor=_FG,
               fontsize=8, loc="upper left")

    #  Panel 5: Per-gate bar chart 
    a = ax[5]
    if gate_labels:
        x     = np.arange(len(gate_labels))
        width = 0.38
        a.bar(x - width/2, g_wall, width, label="Wall (ms)",
              color=_C_WALL, alpha=0.88)
        a.bar(x + width/2, g_cpu,  width, label="CPU (ms)",
              color=_C_RSS,  alpha=0.88)
        a.set_xticks(x)
        a.set_xticklabels(gate_labels, fontsize=7.5, color=_FG, rotation=20, ha="right")
        a.set_ylabel("Time (ms)")
        a.set_title("⑥ Per-gate timing (16q)")
        a.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8)
    else:
        a.text(0.5, 0.5, "No gate records", ha="center", va="center",
               transform=a.transAxes, color=_FG)
        a.set_title("⑥ Per-gate timing (16q)")

    #  Panel 6: Python callable overhead (wide) 
    a = ax[6]

    # Left sub-plot: absolute times
    a.semilogy(q, cpp_pg,  "o-", color=_C_CPP,    lw=2.2, ms=7,
               markerfacecolor=_BG, markeredgewidth=1.8, label="C++ built-in")
    a.semilogy(q, py_ms,   "s-", color=_C_PYLOOP, lw=2.2, ms=7,
               markerfacecolor=_BG, markeredgewidth=1.8, label="Python loop (RY)")
    a.semilogy(q, np_ms,   "^-", color=_C_NUMPY,  lw=2.2, ms=7,
               markerfacecolor=_BG, markeredgewidth=1.8, label="NumPy unitary (H)")
    a.set_xlabel("Qubits"); a.set_ylabel("Wall time per gate (ms)")
    a.set_title("⑦ Python callable overhead vs C++ built-in")
    a.legend(facecolor=_GRID, labelcolor=_FG, fontsize=9)

    # Right y-axis: overhead ratio
    a7b = a.twinx()
    a7b.set_facecolor(_PANEL)
    for sp in a7b.spines.values(): sp.set_edgecolor(_GRID)
    a7b.plot(q, loop_ratio,  "--", color=_C_PYLOOP, lw=1.4, alpha=0.7,
             label="Loop/C++ ratio")
    a7b.plot(q, numpy_ratio, "--", color=_C_NUMPY,  lw=1.4, alpha=0.7,
             label="NumPy/C++ ratio")
    a7b.set_ylabel("Overhead ratio (×)", color=_FG)
    a7b.tick_params(axis="y", colors=_FG, labelsize=8)
    a7b.legend(facecolor=_GRID, labelcolor=_FG, fontsize=8, loc="lower right")

    # Annotate peak overhead
    peak_idx = np.argmax(loop_ratio)
    a7b.annotate(
        f"{loop_ratio[peak_idx]:.0f}× at {q[peak_idx]}q",
        xy=(q[peak_idx], loop_ratio[peak_idx]),
        xytext=(10, 15), textcoords="offset points",
        color=_C_PYLOOP, fontsize=8,
        arrowprops=dict(arrowstyle="->", color=_C_PYLOOP, lw=0.8)
    )

    #  Footer 
    fig.text(
        0.5, 0.01,
        f"qprofiler v2.0  ·  g++ -O3 -march=native -fopenmp  ·  "
        f"State vector: complex128 (16 B/amplitude)  ·  "
        f"Python callable: GIL + zero-copy numpy view  ·  {datetime.now().strftime('%Y-%m-%d')}",
        ha="center", va="bottom", color="#606080", fontsize=8
    )

    #  Save 
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / "profiler_report_v2.png"
    pdf = out_dir / "profiler_report_v2.pdf"
    fig.savefig(png, dpi=155, bbox_inches="tight", facecolor=_BG)
    fig.savefig(pdf,           bbox_inches="tight", facecolor=_BG)
    plt.close(fig)
    print(f"\n  Saved → {png}")
    print(f"  Saved → {pdf}")
    return str(png)


if __name__ == "__main__":
    bench = ROOT / "build" / "qp_bench"
    if not bench.exists():
        print("qp_bench not found — build the C++ binary first.")
        sys.exit(1)
    build_report(str(bench), ROOT / "reports")
