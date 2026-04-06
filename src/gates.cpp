#include "gates.hpp"
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>

#ifdef _OPENMP
# include <omp.h>
#endif

namespace qprofiler {
    
    // Internal helpers

    // Number of amplitudes (state dimension) for n_qubits
    static inline std::size_t dim(int n_qubits) {
        return static_cast<std::size_t>(1) << n_qubits;
    }

    // Bit mask for qubit k in the computational basis index
    static inline std::size_t mask(int k) {
        return static_cast<std::size_t>(1) << k;
    }

    // Hadamard
    // H |0> = (|0> + |1>) / sqrt(2)
    // H |1> = (|0> - |1>) / sqrt(2)

    // For each pair of amplitudes (i, i|mask) where bit k of i is 0:
    // new[i] = (old[i] + old[i|mask]) / sqrt(2)
    // new [i|mask] = (old[i] - old[i|mask]) / sqrt(2)
    void apply_hadamard(StateVec& state, int n_qubits, int target) {
        const std::size_t N = dim(n_qubits);
        const std::size_t m = mask(target);
        const double inv2 = 1.0 / std::sqrt(2.0);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
            for (std::size_t i = 0; i < N; i++) {
                if (i&m) continue; // process only the "lower" of each pair since any i > m will have i&m > 0
                complex_t a = state[i];
                complex_t b = state[i | m];
                state[i] = inv2 * (a + b);
                state[i|m] = inv2 * (a - b);
            }
    }

} // namespace qprofiler