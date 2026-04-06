#include "gates.hpp"
#include <iostream>
#include <iomanip>

int main() {
    const int n_qubits = 1;
    const std::size_t N = 1ULL << n_qubits;

    // Prepare |0> state
    qprofiler::StateVec state(N);
    state[0] = qprofiler::complex_t(1.0, 0.0);
    if (N > 1) state[1] = qprofiler::complex_t(0.0, 0.0);

    // Apply Hadamard to qubit 0
    qprofiler::apply_hadamard(state, n_qubits, 0);

    // Print amplitudes
    std::cout << std::setprecision(12);
    for (std::size_t i = 0; i < state.size(); ++i) {
        std::cout << "amp[" << i << "] = " << state[i] << '\n';
    }

    // Now for |1> state
    qprofiler::StateVec state1(N);
    state1[0] = qprofiler::complex_t(0.0, 0.0);
    state1[1] = qprofiler::complex_t(1.0, 0.0);

    // Applying Hadamard to qubit 1
    qprofiler::apply_hadamard(state1, n_qubits, 0);

    // Print amplitudes
    std::cout << std::setprecision(12);
    for (std::size_t i = 0; i < state1.size(); ++i) {
        std::cout << "amp[" << i << "] = " << state1[i] << '\n';
    }

    return 0;
}
