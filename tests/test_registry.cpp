// Unit tests for Kernel Registry and Simulator::apply_gate()

#include "kernel_registry.hpp"
#include "simulator.hpp"
#include <iostream>
#include <sstream>
#include <cassert>
#include <cmath>
#include <stdexcept>

using namespace qprofiler;

static int passed = 0, failed = 0;

// Assertion macros

#define ASSERT_TRUE(cond, msg)\
    do { if(!(cond)) {std::cerr<<"[FAIL] "<<(msg)<<"\n"; ++failed;}\
         else { ++passed; } } while (0)

#define ASSERT_FALSE(cond, msg) ASSERT_TRUE(!(cond), msg)

# define ASSERT_NEAR(a, b, eps, msg)\
    do { double _d = std::abs((double)(a) - (double)(b));\
         if (_d > (eps)) {\
            std::cerr << "[FAIL] " << (msg) << " got "<<(a)<<" expected "<<(b) <<" (diff="<<_d<<")\n";\
            ++failed;\
        } else ++passed;\
    } while (0)

#define ASSERT_THROWS(expr, ExcType, msg)\
    do { bool _caught = false;\
        try { (void) (expr); }\
        catch (const ExcType&) { _caught = true; }\
        catch (...){}\
        ASSERT_TRUE(_caught, msg);\
    } while(0)

// Helpers

static StateVec zero_state(int n) {
    StateVec s(static_cast<std::size_t>(1)<<n, complex_t(0.0, 0.0));
    s[0] = complex_t(1.0, 0.0);
    return s;
}

static double norm(const StateVec& s) {
    double n2=0;
    for (auto& a: s) n2+=std::norm(a);
    return n2;
}

// Test: register and retrieve a custom kernel
static void test_register_and_retrieve(){
    auto& reg = KernelRegistry::instance();

    bool called = false;
    KernelFn fn = [&called](StateVec&, int, const GateArgs&) {called = true;};
    reg.register_kernel("test_gate_1", fn);
    ASSERT_TRUE(reg.has_kernel("test_gate_1"), "custom kernel registered");

    StateVec s = zero_state(2);
    GateArgs args{{0}, {}};
    reg.get("test_gate_1")(s, 2, args);
    ASSERT_TRUE(called, "custom kernel invoked");
}

// Test: overwrite an existing kernel
static void test_overwrite() {
    auto& reg = KernelRegistry::instance();
    int call_count = 0;
    reg.register_kernel("test_gate_2", [&](StateVec&, int, const GateArgs&){call_count=1;});
    reg.register_kernel("test_gate_2", [&](StateVec&, int, const GateArgs&){call_count=2;});
    StateVec s = zero_state(1);
    GateArgs args{{0}, {}};
    reg.get("test_gate_2")(s, 1, args);
    ASSERT_TRUE(call_count == 2, "second registration overwrites the first");
}

// Test : built in aliases registered at startup
static void test_builtin_aliases() {
    auto& reg = KernelRegistry::instance();

    // All Hadamard aliases
    ASSERT_TRUE(reg.has_kernel("hadamard"), "has hadamard");
    ASSERT_TRUE(reg.has_kernel("h"), "has h alias");
    ASSERT_TRUE(reg.has_kernel("Hadamard"), "has Hadamard alias");

    // Pauli aliases
    ASSERT_TRUE(reg.has_kernel("pauli_x"), "has pauli_x");
    ASSERT_TRUE(reg.has_kernel("x"), "has x alias");
    ASSERT_TRUE(reg.has_kernel("PauliX"), "has PauliX alias");
    ASSERT_TRUE(reg.has_kernel("pauli_z"), "has pauli_z");
    ASSERT_TRUE(reg.has_kernel("z"), "has z alias");
    ASSERT_TRUE(reg.has_kernel("PauliZ"), "has PauliZ alias");

    // CNOT aliases
    ASSERT_TRUE(reg.has_kernel("cnot"), "has cnot");
    ASSERT_TRUE(reg.has_kernel("cx"), "has cx alias");
    ASSERT_TRUE(reg.has_kernel("CNOT"), "has CNOT alias");

    // Phase aliases
    ASSERT_TRUE(reg.has_kernel("phase"), "has phase");
    ASSERT_TRUE(reg.has_kernel("rz"), "has rz");
    ASSERT_TRUE(reg.has_kernel("s"), "has s");
    ASSERT_TRUE(reg.has_kernel("t"), "has t");

    // Unknown gate is not registered
    ASSERT_FALSE(reg.has_kernel("definitely_not_a_gate"), "unknown gate absent");
}

// Test: get() throws for unknown kernel
static void test_get_throws() {
    ASSERT_THROWS(
        KernelRegistry::instance().get("no_such_kernel"), std::out_of_range, "get() throws out_of_range for unknown kernel"
    );
}

// Test: list_kernels() includes built ins
static void test_list_kernels() {
    auto names = KernelRegistry::instance().list_kernels();
    ASSERT_TRUE(names.size() >= 10, "at least 10 built-in kernels listed");
    bool sorted = true;
    for (std::size_t i=1; i<names.size(); ++i) if(names[i] < names[i-1]){sorted=false; break;}
    ASSERT_TRUE(sorted, "list_kernels() returns sorted names");
}

// Test : GateArgs struct passed correctly
static void test_gate_args_forwarding() {
    auto& reg = KernelRegistry::instance();
    std::vector<int> received_targets;
    std::vector<double> received_params;

    reg.register_kernel("test_args_gate", [&](StateVec&, int, const GateArgs& a) {
        received_targets = a.targets;
        received_params = a.params;
    });
    StateVec s = zero_state(4);
    GateArgs args{{1, 3}, {1.23, 4.56}};
    reg.get("test_args_gate")(s, 4, args);

    bool targets_ok = (received_targets == std::vector<int>{1, 3});
    bool params_ok = (received_params == std::vector<double>{1.23, 4.56});
    ASSERT_TRUE(targets_ok, "targets passed successfully");
    ASSERT_TRUE(params_ok, "params forwarded");
}

// Test: Simulator::apply_gate() calls ScopedTimer
static void test_simulator_apply_gate_profiled() {
    Simulator sim(8);
    bool was_called = false;
    Simulator::register_kernel("noop_8q", [&](StateVec&, int, const GateArgs&){was_called = true;});
    sim.apply_gate("noop_8q", {0});
    ASSERT_TRUE(was_called, "apply_gate() invokes the kernel");

    auto& records = sim.profiler().records();
    ASSERT_TRUE(records.size() == 1, "one ProfileRecord emitted");
    ASSERT_TRUE(records[0].label == "noop_8q", "record label = kernel name");
    ASSERT_TRUE(records[0].n_qubits == 8, "record n_qubtis correct");
    ASSERT_TRUE(records[0].wall_ms >= 0.0, "wall_ms non-negative");
}

// Test : apply_gate() throws for bad qu bit index
static void test_apply_gate_bad_qubit() {
    Simulator sim(4);
    ASSERT_THROWS(
        sim.apply_gate("hadamard", {4}), //index 4 out of range [0-3]
        std::out_of_range, "apply_gate throws out_of_range for invalid qubit"
    );
}

// Test: apply_gate() throws for unknown kernel
static void test_apply_gate_unknown_kernel() {
    Simulator sim(4);
    ASSERT_THROWS(sim.apply_gate("unknown_xyz", {0}), std::out_of_range, "apply_gate throws out_of_range for unregistered kernel");
}

// Test : "h" alias is physically identical to "hadamard"
static void test_alias_correctness(){
    StateVec s1 = zero_state(3), s2 = zero_state(3);
    GateArgs a{{0}, {}};

    KernelRegistry::instance().get("hadamard")(s1, 3, a);
    KernelRegistry::instance().get("h")(s2, 3, a);

    for (std::size_t i = 0; i< s1.size(); ++i) {
        ASSERT_NEAR(s1[i].real(), s2[i].real(), 1e-14, "alias real match");
        ASSERT_NEAR(s1[i].imag(), s2[i].imag(), 1e-14, "alias imag match");
    }
}

// Test: S gate (Phase(pi/2)) and T gate (Phase(pi/4))

static void test_s_and_t_gates() {
    auto& reg = KernelRegistry::instance();

    StateVec s1(2, {0, 0}), s2(2, {0, 0});
    s1[1] = 1.0; s2[1] = 1.0;
    GateArgs a{{0}, {}};
    reg.get("s")(s1, 1, a); // S|1> = i|1>
    reg.get("t")(s2, 1, a); // T|1> = (1/sqrt(2))(1+i)|1>
    double inv2 = 1.0 / std::sqrt(2.0);
    ASSERT_NEAR(s1[1].real(),  0.0, 1e-12, "S gate: real part of i");
    ASSERT_NEAR(s1[1].imag(),  1.0, 1e-12, "S gate: imag part of i");
    ASSERT_NEAR(s2[1].real(), inv2, 1e-12, "T gate: real");
    ASSERT_NEAR(s2[1].imag(), inv2, 1e-12, "T gate: imag");
}
         
// Test: parametric kernels receive correct params
static void test_parametric_kernel(){
    // Phase(theta) on |1> makes amplitude e^{i*theta}
    StateVec s(2, {0,0}); s[1] = 1.0;
    GateArgs a{{0}, {M_PI}};  // theta = pi

    KernelRegistry::instance().get("phase")(s, 1, a);

    ASSERT_NEAR(s[1].real(), -1.0, 1e-12, "Phase(pi): real = -1"); // e^{i*pi} = -1
    ASSERT_NEAR(s[1].imag(),  0.0, 1e-12, "Phase(pi): imag = 0");
}


int main() {
    std::cout<<"Running KernelRegistry unit tests...\n\n";

    test_register_and_retrieve();
    test_overwrite();
    test_builtin_aliases();
    test_get_throws();
    test_list_kernels();
    test_gate_args_forwarding();
    test_simulator_apply_gate_profiled();
    test_apply_gate_bad_qubit();
    test_apply_gate_unknown_kernel();
    test_alias_correctness();
    test_s_and_t_gates();
    test_parametric_kernel();

    std::cout<<"\nResults: "<<passed<<" passed, "<<failed<<" failed\n";
    return (failed>0) ? 1: 0;
}
