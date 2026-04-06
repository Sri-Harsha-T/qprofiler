#pragma once
#include <string>
#include <chrono>
#include <cstdint>
#include <vector>
#include <unordered_map>

// Building profile record by monitoring the Resident Set Size (RSS) for processes alongside
// RAII (Resource acquisition is initialization) idiom based profiling with a ScopedTimer which tracks the objects associated 

namespace qprofiler {

    // Timing + Memory Snapshot
    struct ProfileRecord {
        std::string label; // Object label, ex : apply_hadamard
        double wall_ms; // wall-clock time in milliseconds
        double cpu_ms; // cpu time in milliseconds (sums across all OpenMP threads). CPU time > Wall time means parallelism is working
        int64_t peak_rss_kb; // peak resident set size at the END of the call (kB)
        std::size_t state_dim; // 2^n qubits at the time of this call
        int n_qubits;
        std::size_t gate_count; // gates applied in circuit (required for circuit benchmarks)
    };

    // Platform RSS query
    // Returns current process RSS in kB (for Linux: /proc/self/status)
    // Falls to 0 on unsupported platforms
    int64_t current_rss_kb();

    // Returns peak RSS since process start (for Linux: /proc/self/status VmPeak)
    int64_t peak_rss_kb();

    // Scoped Timer with RAII guard.
    // RAII guard writes a ProfileRecord into a collector vector on destruction
    class ScopedTimer {
        public:
            ScopedTimer(
                const std::string& label,
                int n_qubits, 
                std::size_t gate_count,
                std::vector<ProfileRecord>& collector
            );
            ~ScopedTimer();

        private:
            std::string label_;
            int n_qubits_;
            std::size_t gate_count_;
            std::vector<ProfileRecord>& collector_;

            std::chrono::steady_clock::time_point wall_start_;
            std::clock_t cpu_start_;
    };

    // Profiler session
    // Accumulates ProfileRecords and can expose them to Python via Pybind11
    class Profiler {
        public:
            Profiler() = default;

            // Clears all recorded data;
            void reset();

            // Access the raw records (read-only)
            const std::vector<ProfileRecord>& records() const { return records_; }

            // Expose mutable records for ScopedTimer (for convinience)
            std::vector<ProfileRecord>& mutable_records() { return records_; }

            // Prints a human readable summary to stdout
            void print_summary() const;

        private:
            std::vector<ProfileRecord> records_;
    };

} // namespace qprofiler