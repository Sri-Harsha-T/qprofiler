#include "simulator.hpp"
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <string>

// helper functions
static void print_headers() {
    std::cout<<"\n"<<std::left<<std::setw(8)<<"Qubits"<<std::right<<std::setw(14)
             <<"Dim"<<std::setw(14)<<"StateBytes"<<std::setw(12)<<"Wall(ms)"
             <<std::setw(12)<<"CPU(ms)"<<std::setw(14)<<"PeakRSS(kB)"<<std::setw(12)
             <<"Gates"<<std::setw(12)<<"MGates/s"<<"\n"<<std::string(98, '-')<<"\n";
}

static void print_row(const qprofiler::BenchmarkResult& r) {
    std::cout<<std::left<<std::setw(8)<<r.n_qubits<<std::right<<std::setw(14)<<r.state_dim
             <<std::setw(14)<<r.state_bytes<<std::setw(12)<<std::fixed<<std::setprecision(2)
             <<r.wall_ms<<std::setw(12)<<r.cpu_ms<<std::setw(14)<<r.peak_rss_kb<<std::setw(12)
             <<r.gate_count<<std::setw(12)<<std::setprecision(3)<<r.throughput_mgs<<"\n";
}

// Per gate timing demo
// Shows ScopedTimer profiler at individual gate level

static void run_gate_timing_demo(int n_qubits) {
    std::cout<<"\nPer gate timing demo at "<<n_qubits<<" qubits\n";

    qprofiler::Simulator sim(n_qubits);
    int nq = n_qubits;

    sim.hadamard(0);
    sim.pauli_x(1%nq);
    sim.pauli_z(2%nq);
    sim.cnot(0, 1%nq);
    sim.phase(3%nq, M_PI/4.0);
    sim.hadamard(nq-1);
    if (nq > 1) sim.cnot(nq-2, nq-1);
    sim.profiler().print_summary();
}

int main(int argc, char* argv[]) {
    int q_min=4, q_max=20, depth=5;

    if (argc >= 2) q_min = std::atoi(argv[1]);
    if (argc >= 3) q_max = std::atoi(argv[2]);
    if (argc >= 4) depth = std::atoi(argv[3]);

    if (q_min<1||q_max>30||q_min>q_max||depth<1) {
        std::cerr<<"Usage: qp_bench [q_min=4] [q_max=20] [depth=5]\n" << "       q_min in [1,30], q_max <= 30, depth >= 1\n";
        return 1;
    }
    #ifdef _OPENMP
        std::cout<<"OpenMP enabled\n";
    #else 
        std::cout<<"OpenMP disabled (single thread)\n";
    #endif

    std::cout<<"Quantum Profiler Benchmark\n"<<"Qubit range : ["<<q_min<<", "<<q_max<<"]\n"
             <<"Circuit depth: "<<depth<<" layers "<< "(H + CNOT + Phase per layer)\n";

    print_headers();

    for(int q = q_min; q<=q_max;++q) { // Creating a fresh simulator per qubit count so RSS reflets the true peak for that allocation
        qprofiler::Simulator sim(q);
        print_row(sim.run_circuit(depth));
    }

    // per gate demo at a mid-range qubit count
    int demo_q = std::min(16, q_max);
    run_gate_timing_demo(demo_q);

    return 0;
}