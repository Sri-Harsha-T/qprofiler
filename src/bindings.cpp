#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "simulator.hpp"
#include "profiler.hpp"
#include "kernel_registry.hpp"

namespace py = pybind11;
using namespace qprofiler;

// Dict converters to avoid extra dependencies on pybind11 structs
static py::dict record_to_dict(const ProfileRecord& r) {
    py::dict d;
    d["label"] = r.label;
    d["wall_ms"] = r.wall_ms;
    d["cpu_ms"] = r.cpu_ms;
    d["peak_rss_kb"] = r.peak_rss_kb;
    d["n_qubits"] = r.n_qubits;
    d["state_dim"] = r.state_dim;
    d["gate_count"] = r.gate_count;
    return d;
}

static py::dict bench_to_dict(const BenchmarkResult& b) {
    py::dict d;
    d["n_qubits"] = b.n_qubits;
    d["depth"] = b.depth;
    d["wall_ms"] = b.wall_ms;
    d["cpu_ms"] = b.cpu_ms;
    d["peak_rss_kb"] = b.peak_rss_kb;
    d["state_dim"] = b.state_dim;
    d["state_bytes"] = b.state_bytes;
    d["gate_count"] = b.gate_count;
    d["throughput_mgs"] = b.throughput_mgs;
    return d;
}

// Build a C++ KernelFn that calls back into Python with a zero-cpoy state view
// Will use GIL (mutex) for mimickning Meyer's singleton
static KernelFn make_python_kernel(py::object py_callable) {
    return [py_callable](StateVec& state, int n_qubits, const GateArgs& args) {
        py::gil_scoped_acquire gil;

        // Zero copy numpy view: C++ owns the memory, numpy must not free it
        py::array_t<std::complex<double>> state_view(
            { static_cast<py::ssize_t>(state.size())},
            {static_cast<py::ssize_t>(sizeof(complex_t))},
            state.data(),
            py::capsule(state.data(), [](void*) {})
        );
        py_callable(state_view, n_qubits, args.targets, args.params);
    };
}

PYBIND11_MODULE(qprofiler_core, m) {
    m.doc() = R"doc(
        qprofiler_core: quantum gate kernel profiler with Python callable support.
        qprofiler_core
        ==============
        C++ quantum gate-kernel benchmark library exposed to Python via Pybind11.

        Implements a state-vector simulator with profiled gate kernels (Hadamard,
        Pauli-X/Z, CNOT, Phase) and an optional OpenMP multithreading backend.

        Example
        -------
        >>> import qprofiler_core as qc
        >>> sim = qc.Simulator(20)
        >>> result = sim.run_circuit(depth=10)
        >>> print(result["wall_ms"])
    )doc";

    py::module_::import("atexit").attr("register")(
        py::cpp_function([](){
            KernelRegistry::instance().clear_python_kernels();
        })
    );

    m.def("register_kernel",
        [](const std::string& name, py::object callable) {
            if (!PyCallable_Check(callable.ptr())) // check if it is a callable, py::callable doesn't work
                throw py::type_error("register_kernel: second argument must be callable");
            Simulator::register_kernel(name, make_python_kernel(callable));
        },
        py::arg("name"), py::arg("callable"),
        "Register a Python callable as a named, profiled gate kernel.\n\n"
        "Signature: fn(state: np.ndarray, n_qubits: int, targets: list[int], params: list[float]) -> None\n"
        "The state array is a zero-copy view, mutate it in-place.");

    m.def("list_kernels",
        []() {return Simulator::list_kernels();},
        "Return sorted list of all registered kernel names.");

    m.def("has_kernel",
        [](const std::string& name) {
            return KernelRegistry::instance().has_kernel(name);
        },
        py::arg("name"));

    m.def("clear_python_kernels",
        []() {KernelRegistry::instance().clear_python_kernels();},
        "Remove all Python-registered kernels, keeping built-in C++ gates.\n"
        "Called automatically at interpreter shutdown to prevent segfaults\n"
        "from py::object refcount decrements after Py_Finalize().");

    // Simulator
    py::class_<Simulator>(m, "Simulator", R"doc(
        State-vector quantum circuit simulator.

        Parameters
        ----------
        n_qubits : int
            Number of qubits (1-30).  State vector size is 2^n_qubits complex128.
    )doc")
        .def(py::init<int>(), py::arg("n_qubits"))
        .def("reset_state", &Simulator::reset_state, "Reset state vector to |0...0>.")
        .def("hadamard", &Simulator::hadamard, py::arg("target"), "Apply Hadamard gate to qubit `target`")
        .def("pauli_x",  &Simulator::pauli_x,  py::arg("target"))
        .def("pauli_z",  &Simulator::pauli_z,  py::arg("target"))
        .def("cnot",     &Simulator::cnot,     py::arg("ctrl"), py::arg("tgt"))
        .def("phase",    &Simulator::phase,    py::arg("target"), py::arg("theta"))
        .def("apply_gate",
            [](Simulator& self, const std::string& name, const std::vector<int>& targets,
                const std::vector<double>& params) {
                self.apply_gate(name, targets, params);
            },
            py::arg("name"), py::arg("targets"), py::arg("params") = std::vector<double>{},
            "Apply a registered kernel by name. Profiled via ScopedTimer.")
        .def("run_circuit",
             [](Simulator& self, int depth, unsigned seed) -> py::dict {
                 return bench_to_dict(self.run_circuit(depth, seed));
             },
             py::arg("depth"), py::arg("seed") = 42u,
             R"doc(
                 Run a random layered circuit and return a benchmark dict.

                 Keys
                 ----
                 n_qubits, depth, wall_ms, cpu_ms, peak_rss_kb,
                 state_dim, state_bytes, gate_count, throughput_mgs
             )doc")

        .def("sweep_qubits",
             [](Simulator& self, int q_min, int q_max,
                int depth, unsigned seed) -> py::list {
                 py::list out;
                 for (auto& b : self.sweep_qubits(q_min, q_max, depth, seed))
                     out.append(bench_to_dict(b));
                 return out;
             },
             py::arg("q_min"), py::arg("q_max"),
             py::arg("depth"), py::arg("seed") = 42u,
             "Sweep n_qubits from q_min to q_max and return list of benchmark dicts.")

        .def("get_records",
             [](const Simulator& self) -> py::list {
                 py::list out;
                 for (const auto& r : self.profiler().records())
                     out.append(record_to_dict(r));
                 return out;
             },
             "Return list of per-gate ProfileRecord dicts collected by the internal Profiler.")

        .def("print_summary",
             [](const Simulator& self) {
                 self.profiler().print_summary();
             },
             "Print a formatted summary table of profiling records to stdout.")

        .def_property_readonly("n_qubits",   &Simulator::n_qubits)
        .def_property_readonly("state_dim",  &Simulator::state_dim);
    
    // Free function helpers
    m.def("current_rss_kb", &current_rss_kb,
          "Current process RSS in kB (Linux only; 0 elsewhere).");
    m.def("peak_rss_kb",    &peak_rss_kb,
          "Peak process RSS in kB since start (Linux only; 0 elsewhere).");

#ifdef _OPENMP
    m.attr("openmp_enabled") = true;
#else
    m.attr("openmp_enabled") = false;
#endif
}