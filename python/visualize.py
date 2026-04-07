"""
Performance report generator for qprofiler results

Produces a 6 panel PDF/PNG report:
    1. Wall time vs qubit count (linear and log scale)
    2. Peak RSS vs qubit count (with 16B * 2^n theoretical overlay)
    3. Throughput (MGates/s) vs qubit count
    4. CPU vs Wall time ratio (to measure parallelism efficiency)
    5. State vector memory (MB) vs qubits (2^n scaling)
    6. Gate level timing bar chart
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List, Optional
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import LogLocator, AutoMinorLocator
from datetime import datetime

from benchmark import SweepResult, GateRecord

# Colour Pallette (Generated: purple + teal accent)
_C_WALL      = "#7B2FBE"
_C_RSS       = "#00B4D8"
_C_THEORY    = "#E63946"
_C_THRU      = "#06D6A0"
_C_RATIO     = "#F4A261"
_C_STATEMEM  = "#B5179E"
_GRID_COLOR  = "#2A2A3A"
_BG          = "#0F0F1A"
_FG          = "#E0E0F0"

def generate_report(
    sweep_results: List[SweepResult],
    gate_records: Optional[List[GateRecord]] = None,
    output_dir: str = "reports",
    filename: str="profiler_report",
    show: bool=False,
) -> str:
    """
    Generate desired multi panel performance report

    Parameters:
    sweep_results - from benchmark.run_sweep()
    gate_records - from benchmark.profile_gate_sequence() | Optional
    output_dir - Output directory to save PNG and PDF files
    filename - base filename
    show - if True, call plt.show() | Helps visualise in .ipynb notebooks

    Returns Path to the saved PNG file
    """

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Extract Data
    qubits = np.array([r.n_qubits for r in sweep_results])
    wall_ms = np.array([r.wall_ms for r in sweep_results])
    cpu_ms = np.array([r.cpu_ms for r in sweep_results])
    rss_kb = np.array([r.peak_rss_kb for r in sweep_results])
    state_mb = np.array([r.state_mb for r in sweep_results])
    throughput = np.array([r.throughput_mgs for r in sweep_results])
    # py_oh = np.array([r.py_overhead_ms for r in sweep_results])

    theory_mb = (2.0**qubits) * 16.0 / (1024**2) # Theoretical state-vector memory: 2^q * 16 bytes (complex128)
    theory_rss = theory_mb*1024 #kB

    ratio = np.where(wall_ms>0, cpu_ms/wall_ms, 1.0)

    # Figure layout

    fig = plt.figure(figsize=(18, 14), facecolor=_BG)
    fig.suptitle(
        "⚛  Quantum Profiler — Performance Report",
        fontsize=18, color=_FG, fontweight="bold", y=0.98,
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        hspace=0.42, wspace=0.35,
        left=0.07, right=0.97, top=0.92, bottom=0.07,
    )
    axes = [fig.add_subplot(gs[i // 3, i % 3]) for i in range(6)]
    for ax in axes:
        _apply_dark_theme(ax)

    # Panel 1: Wall time (log scale)
    ax = axes[0]
    ax.semilogy(qubits, wall_ms, "o-", color=_C_WALL, lw=2, ms=6, label="Wall time")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Wall Time (ms)")
    ax.set_title("Simulation Wall Time")
    ax.legend(facecolor=_GRID_COLOR, labelcolor=_FG)
    _annotate_scaling(ax, qubits, wall_ms, color=_C_WALL)

    # Panel 2: Peak RSS memory
    ax = axes[1]
    ax.semilogy(qubits, rss_kb,    "s-", color=_C_RSS,    lw=2, ms=6, label="Peak RSS (kB)")
    ax.semilogy(qubits, theory_rss,"--", color=_C_THEORY, lw=1.5, alpha=0.8,
                label="Theoretical 2ⁿ×16B")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Memory (kB)")
    ax.set_title("Peak RSS Memory")
    ax.legend(facecolor=_GRID_COLOR, labelcolor=_FG)

    # Panel 3: Throughput
    ax = axes[2]
    ax.plot(qubits, throughput, "^-", color=_C_THRU, lw=2, ms=7)
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("Throughput (MGates/s)")
    ax.set_title("Gate Throughput vs Qubit Count")
    ax.fill_between(qubits, throughput, alpha=0.15, color=_C_THRU)

    # Panel 4: CPU/Wall ratio (to calculate parallelism efficiency)
    ax = axes[3]
    ax.plot(qubits, ratio, "D-", color=_C_RATIO, lw=2, ms=6)
    ax.axhline(1.0, color=_FG, lw=1, ls=":", alpha=0.5, label="1× (no parallelism)")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("CPU Time / Wall Time")
    ax.set_title("CPU/Wall Ratio (OpenMP Efficiency)")
    ax.legend(facecolor=_GRID_COLOR, labelcolor=_FG)

    # Panel 5: StateVector memory (MB)
    ax = axes[4]
    ax.semilogy(qubits, state_mb,  "o-",  color=_C_STATEMEM, lw=2, ms=6, label="Measured")
    ax.semilogy(qubits, theory_mb, "--",  color=_C_THEORY,   lw=1.5, alpha=0.8,
                label="2^n x 16 B (complex128)")
    ax.set_xlabel("Number of Qubits")
    ax.set_ylabel("State Vector Size (MB)")
    ax.set_title("State-Vector Memory Footprint")
    ax.legend(facecolor=_GRID_COLOR, labelcolor=_FG)

    # Panel 6: Gate level timing bar chart
    ax = axes[5]
    if gate_records:
        labels  = [f"{r.label}\n({r.n_qubits}q)" for r in gate_records]
        w_times = [r.wall_ms   for r in gate_records]
        c_times = [r.cpu_ms    for r in gate_records]
        x       = np.arange(len(labels))
        width   = 0.38

        ax.bar(x - width/2, w_times, width, label="Wall (ms)", color=_C_WALL,   alpha=0.85)
        ax.bar(x + width/2, c_times, width, label="CPU  (ms)", color=_C_RSS,    alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=8, rotation=20, ha="right", color=_FG)
        ax.set_ylabel("Time (ms)")
        ax.set_title("Per-Gate Timing (C++ Profiler)")
        ax.legend(facecolor=_GRID_COLOR, labelcolor=_FG)
    else:
        ax.text(0.5, 0.5, "No gate records.\nCall profile_gate_sequence() first.",
                ha="center", va="center", transform=ax.transAxes, color=_FG, fontsize=12)
        ax.set_title("Per-Gate Timing")

    # Footer annotation
    fig.text(
        0.5, 0.005,
        f"Generated by qprofiler  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}  "
        f"|  State vector: complex128  |  Gates: H, CNOT, Phase",
        ha="center", va="bottom", color="#888888", fontsize=9,
    )

    # Save file
    png_path = os.path.join(output_dir, filename + ".png")
    pdf_path = os.path.join(output_dir, filename + ".pdf")
    fig.savefig(png_path, dpi=150, bbox_inches="tight", facecolor=_BG)
    fig.savefig(pdf_path, bbox_inches="tight", facecolor=_BG)

    if show:
        plt.show()
    plt.close(fig)

    print(f"\nReport saved to {png_path} and {pdf_path}")
    return png_path

# Helper methods

def _apply_dark_theme(ax):
    ax.set_facecolor(_BG)
    ax.tick_params(colors=_FG, which="both")
    ax.xaxis.label.set_color(_FG)
    ax.yaxis.label.set_color(_FG)
    ax.title.set_color(_FG)
    for spine in ax.spines.values():
        spine.set_edgecolor(_GRID_COLOR)
    ax.grid(True, color=_GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.7)

def _annotate_scaling(ax, x, y, color="#AAAAAA"):
    """Fit log(y) ~ k*x and annotate the growth rate k."""
    try:
        valid = y > 0
        if valid.sum() < 3:
            return
        k = np.polyfit(x[valid], np.log2(y[valid]), 1)[0]
        ax.annotate(
            f"  ~2^({k:.2f}n)",
            xy=(x[valid][-1], y[valid][-1]),
            xytext=(-60, 12), textcoords="offset points",
            color=color, fontsize=9, arrowprops=dict(arrowstyle="-", color=color),
        )
    except Exception:
        pass
