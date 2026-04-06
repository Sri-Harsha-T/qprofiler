// Test file to formalize the created gates and profilers

#include "gates.hpp"
#include "profiler.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <string>

using namespace qprofiler;

static int passed=0, failed=0;

#define ASSERT_NEAR(a, b, eps, msg)\
    do {\
        double _diff = std::abs((a) - (b));\
        if (_diff>(eps)) {\
            std::cerr<<"[FAIL] "<<(msg)<<" |" << (a)<<" - " << (b) <<"| = "<<_diff<<"\n";\
            ++failed;\
        } else {\
            ++passed;\
        }\
    } while(0)

#define ASSERT_TRUE(cond, msg)\
    do {\
        if (!(cond)){\
            std::cerr<<"[FAIL] "<<(msg)<<"\n";\
            ++failed;\
        } else {\
            ++passed;\
        }\
    } while(0)

// Test 1 : Hadamard on 1 qubit
// |0> --H-- should give (|0> + |1>) / sqrt(2)
static void test_hadamard_1q() {
    StateVec s = {complex_t(1.0, 0.0), complex_t(0.0, 0.0)};
    apply_hadamard(s, 1, 0);
    double inv2 = 1.0 / std::sqrt(2.0);
    ASSERT_NEAR(s[0].real(), inv2, 1e-12, "H|0>: amplitude[0]");
    ASSERT_NEAR(s[1].real(), inv2, 1e-12, "H|0>: amplitude[1]");

    // H is its own inverse
    apply_hadamard(s, 1, 0);
    ASSERT_NEAR(s[0].real(), 1.0, 1e-12, "H*H|0>: back to |0>");
    ASSERT_NEAR(s[1].real(), 0.0, 1e-12, "H*H|0>: zero amplitude[1]");
}

// Test 2 : CNOT Entanglement
// H on qu bit 0 and then CNOT(ctrl=0, tgt=1) creates Bell state (|00> + |11>)/sqrt(2.0)
static void test_cnot_bell(){ 
    // 2 qubit state: {|00>, |01>, |10>, |11>}
    StateVec s(4, complex_t(0.0, 0.0));
    s[0] = complex_t(1.0, 0.0); // |00>

    apply_hadamard(s, 2, 0);// (|00> + |10>)/sqrt(2)
    apply_cnot(s, 2, 0, 1); // (|00> + |11>)/ sqrt(2)

    double inv2 = 1.0/std::sqrt(2.0);
    ASSERT_NEAR(std::abs(s[0]), inv2, 1e-12, "Bell: |00> amplitude");
    ASSERT_NEAR(std::abs(s[1]), 0.0,  1e-12, "Bell: |01> zero");
    ASSERT_NEAR(std::abs(s[2]), 0.0,  1e-12, "Bell: |10> zero");
    ASSERT_NEAR(std::abs(s[3]), inv2, 1e-12, "Bell: |11> amplitude");

}

// Test 3 : Pauli-X

static void test_pauli_x() {
    StateVec s(2, complex_t(0.0, 0.0));
    s[0] = complex_t(1.0, 0.0);
    apply_pauli_x(s, 1, 0);
    ASSERT_NEAR(s[0].real(), 0.0, 1e-12, "X: flip amplitude 0");
    ASSERT_NEAR(s[1].real(), 1.0, 1e-12, "X: flip amplitude 1");
}

// Test 4 : Phase gate
// Phase(pi) on |1> = -|1> Since e^{i*pi} = -1
static void test_phase_gate() {
    StateVec s = {complex_t(0.0, 0.0), complex_t(1.0, 0.0)};
    apply_phase(s, 1, 0, M_PI);
    ASSERT_NEAR(s[1].real(), -1.0, 1e-12, "Phase(pi): real part");
    ASSERT_NEAR(s[1].imag(),  0.0, 1e-12, "Phase(pi): imag part");

}

// Test 5 : Norm preservation
// After any sequence of unitary gates, the norm must remain 1
static void test_norm_preservation() {
    StateVec s(1<<8, complex_t(0.0, 0.0));
    s[0] = complex_t(1.0, 0.0);
    apply_random_circuit(s, 8, 10, 99);

    double norm2 = 0.0;
    for (const auto& amp: s){
        norm2 += std::norm(amp);
    }

    ASSERT_NEAR(norm2, 1.0, 1e-10, "Norm preservation after random circuit");
}

// Test 6 : RSS query should return non negative value
static void test_rss() {
    int64_t rss = peak_rss_kb();
    // >0 on Linux, 0 elsewhere
    ASSERT_TRUE(rss >= 0, "peak_rss_kb() returns non-negative value");
}

// Test 7: ScopedTimer records entry correctly
static void test_scoped_timer() {
    std::vector<ProfileRecord> records;
    {
        ScopedTimer t("test_op", 5, 3, records);
        // Simulation
        double x = 0;
        for (int i = 0; i<1000;++i) x+=i*0.001;
        (void) x;
    }

    ASSERT_TRUE(records.size() == 1, "ScopedTimer: one record emitted");
    ASSERT_TRUE(records[0].label == "test_op", "ScopedTimer: label correct");
    ASSERT_TRUE(records[0].n_qubits == 5, "ScopedTimer: n_qubits correct");
    ASSERT_TRUE(records[0].gate_count == 3, "ScopedTimer: gate_count correct");
    ASSERT_TRUE(records[0].wall_ms >= 0.0, "ScopedTimer: wall_ms non-negative");
}

int main() {
    std::cout<<"Quantum Profiler unit tests starting...\n";

    test_hadamard_1q();
    test_cnot_bell();
    test_pauli_x();
    test_phase_gate();
    test_norm_preservation();
    test_rss();
    test_scoped_timer();

    std::cout<<"\nResults: "<< passed<<" passed, "<<failed<<" failed\n";
    return (failed>0);
}