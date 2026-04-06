#pragma once
#include "gates.hpp"
#include "profiler.hpp"
#include <string>
#include <vector>

namespace qprofiler {

    // Holds per-run statistics which can be exported to Python
    struct BenchmarkResult {
        int n_qubits;
        int depth;
        double wall_ms;
        double cpu_ms;
        int64_t peak_rss_kb;
        std::size_t state_dim;  // 2^n_qubits
        std::size_t state_bytes; // memory for one StateVec
        std::size_t gate_count;
        double throughput_mgs; // million gate steps per second
    };

    // Simulator 
    class Simulator {
        public:
            explicit Simulator(int n_qubits); // for direct initialization

            // Resets state to |00...0>
            void reset_state();

            // Single gate wrappers (profiled individually)
            void hadamard(int target);
            void pauli_x(int target);
            void pauli_z(int target);
            void cnot(int ctrl, int tgt);
            void phase(int target, double theta);

            // Run a full random circuit and return the BenchmarkResult for it
            BenchmarkResult run_circuit(int depth, unsigned seed = 42);

            // Sweep n_qubits from `q_min` to `q_max` for `depth` layers each
            // Returns list of BenchmarkResult (one per qubit count)
            std::vector<BenchmarkResult> sweep_qubits(int q_min, int q_max, int depth, unsigned seed=42);

            int n_qubits() const {return n_qubits_;}
            std::size_t state_dim() const {return state_.size();}

            // Access internal profiler (read-only from Python)
            const Profiler& profiler() const {return prof_;}
            Profiler& profiler() {return prof_;}

        private:
            int n_qubits_;
            StateVec state_;
            Profiler prof_;

            void check_qubit(int q, const std::string& ctx) const;
    };

} // namespace qprofiler