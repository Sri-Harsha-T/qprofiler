#include "simulator.hpp"
#include <stdexcept>
#include <sstream>
#include <cstring>



namespace qprofiler {

    Simulator::Simulator(int n_qubits) : n_qubits_(n_qubits) {
        const int min_qubits = 1, max_qubits = 30;
        if (n_qubits < min_qubits || n_qubits > max_qubits) {
            throw std::invalid_argument("n_qubits must be in [1,30]");
        }
        state_.assign(static_cast<std::size_t>(1)<<n_qubits, complex_t(0.0, 0.0));
        state_[0] = complex_t(1.0, 0.0); // |00....0>
    }

    void Simulator::reset_state() {
        std::fill(state_.begin(), state_.end(), complex_t(0.0, 0.0));
        state_[0] = complex_t(1.0, 0.0); // back to |0...0> state
    }

    void Simulator::check_qubit(int q, const std::string& ctx) const {
        if (q<0 || q>=n_qubits_) {
            std::ostringstream oss;
            oss << ctx <<": qubit "<<q<< " out of range [0, "<<n_qubits_ - 1 << "]";
            throw std::out_of_range(oss.str());
        }
    }

    void Simulator::hadamard(int target) {
        check_qubit(target, "hadamard");
        ScopedTimer t("hadamard", n_qubits_, 1, prof_.mutable_records());
        apply_hadamard(state_, n_qubits_, target);
    }

    void Simulator::pauli_x(int target) {
        check_qubit(target, "pauli_x");
        ScopedTimer t("pauli_x", n_qubits_, 1, prof_.mutable_records());
        apply_pauli_x(state_, n_qubits_, target);
    }

    void Simulator::pauli_z(int target) {
        check_qubit(target, "pauli_z");
        ScopedTimer t("pauli_z", n_qubits_, 1, prof_.mutable_records());
        apply_pauli_z(state_, n_qubits_, target);
    }

    void Simulator::cnot(int ctrl, int tgt) {
        check_qubit(ctrl, "cnot:ctrl");
        check_qubit(tgt, "cnot:tgt");
        ScopedTimer t("cnot", n_qubits_, 1, prof_.mutable_records());
        apply_cnot(state_, n_qubits_, ctrl, tgt);
    }

    void Simulator::phase(int target, double theta) {
        check_qubit(target, "phase");
        ScopedTimer t("phase", n_qubits_, 1, prof_.mutable_records());
        apply_phase(state_, n_qubits_, target, theta);
    }

    BenchmarkResult Simulator::run_circuit(int depth, unsigned seed) {
        reset_state();

        auto t0_wall = std::chrono::steady_clock::now();
        auto t0_cpu = std::clock();

        std::size_t gates = apply_random_circuit(state_, n_qubits_, depth, seed);

        auto t1_wall = std::chrono::steady_clock::now();
        auto t1_cpu = std::clock();

        double wall_ms = std::chrono::duration<double, std::milli>(t1_wall - t0_wall).count();
        double cpu_ms = 1000.0 * static_cast<double>(t1_cpu - t0_cpu) / CLOCKS_PER_SEC;

        std::size_t sdim = state_.size();
        std::size_t sbytes = sdim * sizeof(complex_t);

        double throughput = (wall_ms>0.0) ? (static_cast<double>(gates)/(wall_ms*1e-3)) / 1e6 : 0.0;

        BenchmarkResult res;
        res.n_qubits = n_qubits_;
        res.depth = depth;
        res.wall_ms = wall_ms;
        res.cpu_ms = cpu_ms;
        res.peak_rss_kb = peak_rss_kb();
        res.state_dim = sdim;
        res.state_bytes = sbytes;
        res.gate_count = gates;
        res.throughput_mgs = throughput;

        return res;
    }

    std::vector<BenchmarkResult> Simulator::sweep_qubits(int q_min, int q_max, int depth, unsigned seed) {
        std::vector<BenchmarkResult> results;
        results.reserve(q_max - q_min + 1);

        for (int q = q_min; q <= q_max; ++q) {
            // Rebuilding simulator for each qubit count so that memory reflects true peak
            Simulator tmp(q);
            results.push_back(tmp.run_circuit(depth, seed));
        }
        return results;
    }

} // namespave qprofiler