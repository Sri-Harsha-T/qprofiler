#pragma once
#include "gates.hpp"
#include "profiler.hpp"
#include "kernel_registry.hpp"
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

            // Plugin / callable kernel API

            // Register a KernelFn under `name` in the global KernelRegistry
            // Once registered, the kernel can be invoked via apply_gate()
            static void register_kernel(const std::string& name, KernelFn fn);

            // Apply any registered kernel by name(built-in/user provided)
            // Wraps the call in ScopedTimer, profiled identically to built-ins
            /// @param name Kernel name (must be in KernelRegistry)
            /// @param targets Qubit indices the gate acts on
            /// @param params Rotation angles/phases (empty for Clifford gates)
            void apply_gate(const std::string& name, const std::vector<int>& targets, const std::vector<double>& params = {}); 

            // List all registered kernel names (built-in + user provided)
            static std::vector<std::string> list_kernels();

            // Circuit level benchmarking
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