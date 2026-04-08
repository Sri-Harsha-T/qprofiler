#pragma once
#include "gates.hpp"
#include <functional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <stdexcept>

namespace qprofiler {

    // Gate argument bundle
    // Passed to every Kernel Function to keep the function signature uniform regardless
    // of whether the gate is single-qu bit, two-qubit or parametric
    struct GateArgs {
        std::vector<int> targets; // qubit indices in the state vector
        std::vector<double> params; // rotation angles, phases, etc.
    };

    // KernelFn type
    // Any function (C++ lambda, pybind11 wrapped Python callable, dlopen symbol) that mutates statevector in place
    using KernelFn = std::function<void(StateVec&, int /*n_qubits*/, const GateArgs&)>;

    // KernelRegistry
    // Thread safe-to-read singleton. Kernels are registered at 
    // startup or at Python level via qprofiler_core.register_kernel()
    class KernelRegistry {
        public:
            // Meyers singleton - safe, zero-overhead after first call
            static KernelRegistry& instance();

            //Register (or overwrite) kernel by name
            void register_kernel(const std::string& name, KernelFn fn);

            // True if a kernel has been registered under this name
            bool has_kernel(const std::string& name) const;

            // Retrieve a kernel, throws std::out_of_range if not found
            const KernelFn& get(const std::string& name) const;

            // List all registered kernel names
            std::vector<std::string> list_kernels() const;

            // Remove all registrations 
            void clear();

            // Remove only the entries that were registered after construction
            // i.e. Python-callable kernels, leaving the builtin C++ kernels intact.
            // Safe to call at any time during normal execution
            void clear_python_kernels();

        private:
            KernelRegistry(); // seeds built-in kernels

            std::unordered_map<std::string, KernelFn> table_;
            std::unordered_set<std::string> builtins_; // names seeded in constructor
    };

} // namespace qprofiler