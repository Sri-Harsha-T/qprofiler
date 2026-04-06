#include "profiler.hpp"
#include <ctime>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <iomanip>
#include <algorithm>

#if defined(__linux__)
#include <sys/resource.h>
#endif

namespace qprofiler {

    // RSS (Resident Set Size) calculation helpers

    int64_t current_rss_kb() {
        #if defined(__linux__)
            // Read VmRSS from /proc/self/status
            FILE* fp = std::fopen("/proc/self/status", "r");
            if (!fp) return 0;
            char line[128];
            int64_t rss = 0;
            while (std::fgets(line, sizeof(line), fp)) {
                if (std::strncmp(line, "VmRSS:", 6) == 0) {
                    std::sscanf(line+6, "%lld", (long long*)&rss);
                    break;
                }
            }
            std::fclose(fp);
            return rss;
        #else 
            return 0; // For unsupported platform
        #endif
    }

    int64_t peak_rss_kb() {
        #if defined(__linux__)
            // Read VmPeak from /proc/self/status
            FILE* fp = std::fopen("/proc/self/status", "r");
            if (!fp) return 0;
            char line[128];
            int64_t peak = 0;
            while (std::fgets(line, sizeof(line), fp)) {
                if (std::strncmp(line, "VmPeak:", 7) == 0) {
                    std::sscanf(line+7, "%lld", (long long*)&peak);
                    break;
                }
            }
            std::fclose(fp);
            return peak;
        #else
            return 0;
        #endif
    }

    // ScopedTimer methods

    ScopedTimer::ScopedTimer(
        const std::string& label, 
        int n_qubits, 
        std::size_t gate_count, 
        std::vector<ProfileRecord>& collector
    ) : label_(label), n_qubits_(n_qubits), gate_count_(gate_count), collector_(collector),
        wall_start_(std::chrono::steady_clock::now()),
        cpu_start_(std::clock()) {}

    ScopedTimer::~ScopedTimer() {
        auto wall_end = std::chrono::steady_clock::now();
        auto cpu_end = std::clock();

        double wall_ms = std::chrono::duration<double, std::milli>(wall_end - wall_start_).count();
        double cpu_ms = 1000.0 * static_cast<double>(cpu_end - cpu_start_) / CLOCKS_PER_SEC;

        ProfileRecord p_rec;
        p_rec.label = label_;
        p_rec.wall_ms = wall_ms;
        p_rec.cpu_ms = cpu_ms;
        p_rec.peak_rss_kb = peak_rss_kb();
        p_rec.n_qubits = n_qubits_;
        p_rec.state_dim = (static_cast<std::size_t>(1) << n_qubits_);
        p_rec.gate_count = gate_count_;

        collector_.push_back(p_rec);
    }

    // Profiler methods

    void Profiler::reset() {
        records_.clear();
    }

    void Profiler::print_summary() const {
        if (records_.empty()) {
            std::cout << "[Profiler] No records.\n";
            return ;
        }
        constexpr int w = 24;
        std::cout << "\n" << std::left<<std::setw(w)<<"Label"<<std::right
                   << std::setw(8) << "Qubits"
                   << std::setw(12) << "Wall(ms)"
                   << std::setw(12) << "CPU(ms)"
                   << std::setw(14) << "PeakRSS(kB)"
                   << std::setw(12) << "Gates" << "\n" << std::string(w + 8 + 12 + 12 + 14 + 12, '-')<<"\n";
        
        for (const auto& r: records_) {
            std::cout << std::left << std::setw(w) << r.label
                      << std::right << std::setw(8) << r.n_qubits
                      << std::setw(12) << std::fixed << std::setprecision(3) << r.wall_ms
                      << std::setw(12) << r.cpu_ms << std::setw(14) << r.peak_rss_kb
                      << std::setw(12) << r.gate_count << "\n";
        }
        std::cout<<"\n";
                
    }

} // namespace qprofiler