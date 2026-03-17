"""
setup.py – build the guided_desc_match C++ extension via CMake and install it
as the `epipolar_matching` Python package.

Usage:
    pip install -e .          # editable install (recommended for development)
    pip install .             # regular install
    python setup.py build_ext # build only (output → epipolar_matching/)

Prerequisites (must be on PATH / findable by CMake):
    cmake >= 3.14
    C++17 compiler  (MSVC 2019+ on Windows, GCC/Clang on Linux/macOS)
    pybind11        (pip install pybind11)
    Eigen3          (conda: conda install -c conda-forge eigen)
    OpenCV          (pip install opencv-python  OR  conda install -c conda-forge opencv)
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name: str, sourcedir: str = "") -> None:
        super().__init__(name, sources=[])
        self.sourcedir = str(Path(sourcedir).resolve())


class CMakeBuild(build_ext):
    def build_extension(self, ext: CMakeExtension) -> None:
        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Let pybind11 installed via pip be discoverable by CMake
        cmake_args = [
            f"-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
        ]

        # Point CMake to the pybind11 cmake dir if installed via pip
        try:
            import pybind11
            cmake_args.append(f"-Dpybind11_DIR={pybind11.get_cmake_dir()}")
        except ImportError:
            pass  # rely on system/conda pybind11

        build_args = ["--config", "Release", "--parallel"]

        print(f"\n── Configuring CMake in {build_temp} ──")
        subprocess.run(
            ["cmake", ext.sourcedir, *cmake_args],
            cwd=build_temp,
            check=True,
        )

        print("\n── Building ──")
        subprocess.run(
            ["cmake", "--build", str(build_temp), *build_args],
            check=True,
        )

        print("\n── Done.  Module written to epipolar_matching/ ──\n")


setup(
    name="epipolar_matching",
    version="0.1.0",
    description="Pixel-accurate epipolar guided matching with segment tree",
    long_description=(Path(__file__).parent / "README.md").read_text(encoding="utf-8"),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
    install_requires=["numpy"],
    packages=["epipolar_matching"],
    # Include the pre-built .pyd / .so that CMake places in epipolar_matching/
    package_data={"epipolar_matching": ["*.pyd", "*.so", "*.dylib"]},
    ext_modules=[CMakeExtension("guided_desc_match", sourcedir="cpp")],
    cmdclass={"build_ext": CMakeBuild},
)
