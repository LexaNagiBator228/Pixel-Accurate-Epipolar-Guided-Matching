#!/usr/bin/env bash
set -euo pipefail

echo "============================================================"
echo " Building guided_desc_match Python extension (Linux/macOS)"
echo "============================================================"

# Create output package directory
mkdir -p epipolar_matching

BUILD_DIR="cpp/build"
mkdir -p "$BUILD_DIR"

# Detect pybind11 cmake dir from pip installation
PYBIND11_ARGS=""
if python3 -c "import pybind11" 2>/dev/null; then
    PYBIND11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
    PYBIND11_ARGS="-Dpybind11_DIR=${PYBIND11_DIR}"
fi

echo ""
echo "[1/2] Configuring CMake..."
cmake cpp -B "$BUILD_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python3)" \
    $PYBIND11_ARGS

echo ""
echo "[2/2] Building..."
cmake --build "$BUILD_DIR" --config Release --parallel "$(nproc)"

echo ""
echo "============================================================"
echo " Build succeeded.  Module is in epipolar_matching/"
echo " Run the demo:  python3 demo.py"
echo "============================================================"
