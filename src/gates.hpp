#pragma once
#include <vector>
#include <complex>
#include <cstdint>
#include <string>

namespace qprofiler {

    using complex_t = std::complex<double>;
    using StateVec = std::vector<complex_t>;


    // Gate application kernels
    // Each function mutates the state vector in-place
    // State vector has 2^n entries for n qubits

    // Apply Hadamard gate to qubit `target` in a state vector of `n_qubits`
    void apply_hadamard(StateVec& state, int n_qubits, int target);

} // namespace qprofiler
