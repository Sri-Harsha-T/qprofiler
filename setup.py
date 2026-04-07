"""
pip-installable build for qprofiler using pybind11

Usage : 
pip install pybind11 numpy
pip install -e . # editable install (for development)

Or use the CMake workflow (recommended for OpenMP support):
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
"""

import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

class CMakeBuild(build_ext):
    """Custom build step that delegates to CMake"""

    def run(self):
        for ext in self.extensions:
            self.build_cmake(ext)

    def build_cmake(self, ext):
        cwd = Path(__file__).parent.resolve()
        build_dir = cwd / "build"
        build_dir.mkdir(exist_ok=True)
        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={cwd / 'python'}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DCMAKE_BUILD_TYPE=Release",
        ]

        build_args = ["--build", ".", "--target", "qprofiler_core", "-j4"]

        subprocess.run(
            ["cmake", str(cwd)] + cmake_args,
            cwd=str(build_dir), check=True
        )
        subprocess.run(
            ["cmake"] + build_args,
            cwd=str(build_dir), check=True
        )

setup(
    name                  = "qprofiler",
    version               = "1.0.0",
    author                = "Quantum Profiler | Sri Harsha",
    description           = "Performance profiler for quantum circuit simulations (C++/Python)",
    long_description      = (Path(__file__).parent / "README.md").read_text(),
    long_description_content_type = "text/markdown",
    packages              = find_packages(where="python"),
    package_dir           = {"": "python"},
    python_requires       = ">=3.10",
    extras_require        = {
        "dev": ["pytest>=7.0", "line-profiler"],
    },
    ext_modules           = [Extension("qprofiler_core", sources=[])],
    cmdclass              = {"build_ext": CMakeBuild},
    zip_safe              = False,
    classifiers           = [
        "Programming Language :: Python :: 3",
        "Programming Language :: C++",
    ],
)