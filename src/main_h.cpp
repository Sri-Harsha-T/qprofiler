#include "gates.hpp"
#include <iostream>
#include <iomanip>

int main() {
    const int n_qubits = 2;
    const std::size_t N = 1ULL << n_qubits;

    // Creating a bell state by applying Hadamard on qubit 0 and CNOT from 0 to 1
    // Preparing |00> state
    qprofiler::StateVec state(N, qprofiler::complex_t(0.0, 0.0));
    state[0] = qprofiler::complex_t(1.0, 0.0); // |00>

    // Applying Hadamard to qubit 0
    qprofiler::apply_hadamard(state, n_qubits, 0); // (|00> + |10>)/sqrt(2)
    // Applying CNOT now
    qprofiler::apply_cnot(state, n_qubits, 0, 1); // (|00> + |11>) / sqrt(2)

    // Printing amplitudes
    std::cout<<std::setprecision(12);
    for (std::size_t i = 0; i < state.size(); ++i) {
        std::cout << "amp["<<i<<"] = " << state[i] << "\n";
    }

    return 0;
}
