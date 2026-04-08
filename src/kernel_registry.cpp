#include "kernel_registry.hpp"
#include <stdexcept>
#include <algorithm>

namespace qprofiler {

    // Singleton (Meyer's)
    KernelRegistry& KernelRegistry::instance() {
        static KernelRegistry inst;
        return inst;
    }

    // Constructor for KernelRegistry : seed all built in gate kernels
    // Each built-in lambda adapts the (StateVec&, n_qubits, GateArgs) signature to the 
    // existing apply_* functions. This means all existing gate methods on Simulator keep
    // working unchanged, they just go through the registry now
    KernelRegistry::KernelRegistry() {
        // Hadamard
        table_["hadamard"] = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("hadamard: no target");
            apply_hadamard(s, n, a.targets[0]);
        };

        // Pauli X aliases: "pauli_x", "x", "PauliX" (PennyLane), "x" (Qiskit)
        auto x_fn = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("pauli_x: no target");
            apply_pauli_x(s, n, a.targets[0]);
        };
        table_["pauli_x"] = x_fn;
        table_["x"] = x_fn;
        table_["PauliX"] = x_fn;

        // Pauli Z aliases
        auto z_fn = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("pauli_z: no target");
            apply_pauli_z(s, n, a.targets[0]);
        };
        table_["pauli_z"] = z_fn;
        table_["z"] = z_fn;
        table_["PauliZ"] = z_fn;

        // Hadamard aliases
        auto h_fn = table_["hadamard"];
        table_["h"] = h_fn;
        table_["Hadamard"] = h_fn;

        // CNOT aliases
        auto cx_fn = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.size() < 2) throw std::invalid_argument("cnot: need 2 targets");
            apply_cnot(s, n, a.targets[0], a.targets[1]);
        };
        table_["cnot"] = cx_fn;
        table_["cx"] = cx_fn;
        table_["CNOT"] = cx_fn;

        // Phase and RZ family
        // Phase(theta) : diag(1, e^{i*theta})
        table_["phase"] = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("phase: no target");
            double theta = a.params.empty() ? 0.0 : a.params[0];
            apply_phase(s, n, a.targets[0], theta);
        };

        // RZ(theta): diag(e^{-i*theta/2}, e^{i*theta/2}) by convention
        table_["rz"] = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("rz: no target");
            double theta = a.params.empty() ? 0.0 : a.params[0];
            // RZ = e^{-i*theta/2}*(Phase(theta)) global phase is ignored
            apply_phase(s, n, a.targets[0], theta);
        };
        table_["RZ"] = table_["rz"];

        // S gate = Phase(pi/2), T gate = Phase(pi/4)
        table_["s"] = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("s: no target");
            apply_phase(s, n, a.targets[0], M_PI/2.0);
        };
        table_["S"] = table_["s"];

        table_["t"] = [](StateVec& s, int n, const GateArgs& a) {
            if (a.targets.empty()) throw std::invalid_argument("t: no target");
            apply_phase(s, n, a.targets[0], M_PI/4.0);
        };
        table_["T"] = table_["t"];

        // Record every name seeded in this constructor as a builtin
        // clear_python_kernels() will preserve these and only erase the rest.
        for (const auto& kv: table_)
            builtins_.insert(kv.first);
    }

    // API

    void KernelRegistry::register_kernel(const std::string& name, KernelFn fn) {
        if (!fn) throw std::invalid_argument("register_kernel: null kernel for '" + name + "'");
        table_[name] = std::move(fn);
    }

    bool KernelRegistry::has_kernel(const std::string& name) const {
        return table_.count(name) > 0;
    }

    const KernelFn& KernelRegistry::get(const std::string& name) const {
        auto it = table_.find(name);
        if (it == table_.end()) throw std::out_of_range("KernelRegistry: no kernel named '" + name + "'. Call register_kernel() first");
        return it->second;
    }

    std::vector<std::string> KernelRegistry::list_kernels() const {
        std::vector<std::string> names;
        names.reserve(table_.size());
        for (const auto& kv: table_) names.push_back(kv.first);
        std::sort(names.begin(), names.end());
        return names;
    }

    void KernelRegistry::clear() {
        table_.clear();
    }

    void KernelRegistry::clear_python_kernels() {
        // Erase every entry that is NOT a built-in
        // This drops all captured py::object references while the interpreter
        // is still alive, preventing segfault on shutdown
        for (auto it = table_.begin(); it != table_.end(); ) {
            if (builtins_.count(it->first) == 0)
                it = table_.erase(it);
            else
                ++it;
        }
    }

} // namespace qprofiler