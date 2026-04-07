"""
Py tests for profiler framework
"""

import sys, os
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

# Import C++ library
try:
    import qprofiler_core as qc
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False

skip_no_cpp = pytest.mark.skipif(
    not CPP_AVAILABLE,
    reason="C++ extension not built. Run cmake -B build && cmake --build build"
)

# C++ extension tests

@skip_no_cpp
class TestSimulatorBasic:
    def test_instantiation(self):
        sim = qc.Simulator(5)
        assert sim.n_qubits == 5
        assert sim.state_dim == 32

    def test_invalid_qubits(self):
        with pytest.raises(Exception):
            qc.Simulator(0)
        with pytest.raises(Exception):
            qc.Simulator(31)

    def test_run_circuit_returns_dict(self):
        sim = qc.Simulator(8)
        res = sim.run_circuit(depth=3)
        assert isinstance(res, dict)
        for key in ("n_qubits", "depth", "wall_ms", "cpu_ms", "peak_rss_kb", "state_dim", "state_bytes", "gate_count", "throughput_mgs"):
            assert key in res # , f"Missing key: {key}"

    def test_wall_time_positive(self):
        sim = qc.Simulator(10)
        res = sim.run_circuit(depth=5)
        assert res["wall_ms"] >= 0.0

    def test_state_dim_power_of_two(self):
        for q in [4, 8, 12, 16]:
            sim = qc.Simulator(q)
            res = sim.run_circuit(depth=1)
            assert res["state_dim"] == 2**q

    def test_state_bytes_correct(self):
        q = 10
        sim = qc.Simulator(q)
        res = sim.run_circuit(depth=1)
        expected = (2**q) * 16 # complex128 = 16bytes
        assert res["state_bytes"] == expected

    def test_gate_count_positive(self):
        sim = qc.Simulator(6)
        res = sim.run_circuit(depth=4)
        assert res["gate_count"] > 0

    def test_sweep_returns_list(self):
        sim = qc.Simulator(4)
        results = sim.sweep_qubits(q_min=4, q_max=8, depth=2)
        assert isinstance(results, list)
        assert len(results) == 5 # 4,...,8

    def test_sweep_monotone_state_dim(self):
        sim = qc.Simulator(4)
        results = sim.sweep_qubits(q_min=4, q_max=10, depth=2)
        dims = [r["state_dim"] for r in results]
        assert dims == sorted(dims), "State dims should increase with qubits"

    def test_get_records_after_gates(self):
        sim = qc.Simulator(5)
        pi = 3.141592653
        sim.hadamard(0)
        sim.cnot(0, 1)
        sim.phase(2, pi/2.0)
        records = sim.get_records()
        assert len(records) == 3
        labels = [r["label"] for r in records]
        assert "hadamard" in labels
        assert "cnot" in labels
        assert "phase" in labels

    def test_individual_gate_out_of_range(self):
        sim = qc.Simulator(4)
        with pytest.raises(Exception):
            sim.hadamard(4) # qubit index exceeds the 0<=idx<n_qubits range

    def test_rss_non_negative(self):
        assert qc.current_rss_kb() >= 0
        assert qc.peak_rss_kb() >=0

    def test_openmp_flag_is_bool(self):
        assert isinstance(qc.openmp_enabled, bool)

# Python benchmark tests, no cpp
from benchmark import run_sweep, profile_gate_sequence
from visualize import generate_report

class TestSweepResult:
    """Testing the SweepResult dataclass"""

    @skip_no_cpp
    def test_run_sweep_returns_list(self):
        results = run_sweep(q_min=4, q_max=8, depth=2, verbose=False)
        assert len(results) == 5

    @skip_no_cpp
    def test_sweep_state_mb_grows(self):
        results = run_sweep(q_min=6, q_max=12, depth=2, verbose=False)
        mbs = [r.state_mb for r in results]
        assert mbs == sorted(mbs)

    @skip_no_cpp
    def test_profile_gate_sequence(self):
        records = profile_gate_sequence(n_qubits=8)
        assert len(records) > 0
        for r in records:
            assert hasattr(r, "label")
            assert hasattr(r, "wall_ms")
            assert r.wall_ms >= 0

# Visualize module tests
class TestVisualize:
    @skip_no_cpp
    def test_generate_report_creates_file(self, tmp_path):
        results = run_sweep(q_min = 4, q_max = 8, depth = 2, verbose = False)
        png_path = generate_report(sweep_results=results, output_dir= str(tmp_path), filename="test_report")
        assert os.path.exists(png_path)
        assert os.path.getsize(png_path) > 0

