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

    // Pauli X 
    // X |0> = |1> and vice versa
    void apply_pauli_x(StateVec& state, int n_qubits, int target) {
        const std::size_t N = dim(n_qubits);
        const std::size_t m = mask(target);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < N; ++i) {
            if (i&m) continue;
            std::swap(state[i], state[i|m]); // swaps a|0> + b|1> to a|1> + b|0>
        }
    }

    // Pauli Z
    // Z |0> = |0> but Z|1> = -|1> 
    void apply_pauli_z(StateVec& state, int n_qubits, int target) {
        const std::size_t N = dim(n_qubits);
        const std::size_t m = mask(target);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < N; ++i){
            if (i&m) state[i] = -state[i]; // inverting the phase about the z-axis
        }
    }

    // CNOT
    // CNOT |ctrl=1, tgt=0> = |ctrl=1, tgt=1> and vice-versa
    void apply_cnot(StateVec& state, int n_qubits, int ctrl, int tgt) {
        // ensure that control and target qubtis are not the same
        if(ctrl==tgt){
            throw std::invalid_argument("CNOT: control and target must differ");
        }

        const std::size_t N = dim(n_qubits);
        const std::size_t mc = mask(ctrl);
        const std::size_t mt = mask(tgt);

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < N; ++i){
            if ((i&mc) && !(i&mt)){ // acts only when control qubit is |1>
                std::swap(state[i], state[i|mt]);
            }
        }
    }

    // Phase gate : diag(1, e^{i*theta})
    void apply_phase(StateVec& state, int n_qubits, int target, double theta) {
        const std::size_t N = dim(n_qubits);
        const std::size_t m = mask(target);
        const complex_t ph = std::exp(complex_t(0.0, theta));

        #ifdef _OPENMP
        #pragma omp parallel for schedule(static)
        #endif
        for (std::size_t i = 0; i < N; ++i) {
            if (i&m) state[i] *= ph; // multiply |1> amplitudes with e^(i*theta)
        }
    }

    // Random circuit generator with multiple layers of Hadarmard and entangling CNOT pairs along with random phase rotations
    std::size_t apply_random_circuit(StateVec& state, int n_qubits, int depth, unsigned seed) {
        std::mt19937 rng(seed); // rng generator
        std::uniform_int_distribution<> qubit_dist(0, n_qubits-1);
        std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI); // for phase rotations
        
        std::size_t gate_count = 0;

        for (int layer = 0; layer < depth; ++layer) {
            // Single qubit Hadamard pass
            for (int q = 0; q < n_qubits; ++q) {
                apply_hadamard(state, n_qubits, q);
                ++gate_count;
            }

            // CNOT pairs (Entangling qubits here)
            for (int q = 0; q < n_qubits - 1; q+=2){
                apply_cnot(state, n_qubits, q, q+1);
                ++gate_count;
            }

            // Random phase rotations
            for (int q = 0; q < n_qubits; ++q) {
                apply_phase(state, n_qubits, q, angle_dist(rng));
                ++gate_count;
            }
        }
        return gate_count;
    }

} // namespace qprofiler