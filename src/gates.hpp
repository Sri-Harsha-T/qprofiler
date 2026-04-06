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

    // Apply Pauli X (NOT) gate to qubit `target`
    void apply_pauli_x(StateVec& state, int n_qubits, int target);

    // Apply Pauli Z gate to qubit `target`
    void apply_pauli_z(StateVec& state, int n_qubits, int target);

    // Apply CNOT gate from control qubit `ctrl` to target qubit `tgt`
    void apply_cnot(StateVec& state, int n_qubits, int ctrl, int tgt);

    // Apply a single qubit phase gate with angle theta (diag(1, e^{i*theta})) to `target` qubit
    void apply_phase(StateVec& state, int n_qubits, int target, double theta); 

    // Create a random_circuit of `depth` layers, each consisting of Hadamard + CNOT entangling gates across all qubits
    // Returns total number of gate operations applied
    std::size_t apply_random_circuit(StateVec& state, int n_qubits, int depth, unsigned seed = 42);

} // namespace qprofiler
