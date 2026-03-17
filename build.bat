@echo off
setlocal EnableDelayedExpansion

echo ============================================================
echo  Building guided_desc_match Python extension (Windows)
echo ============================================================

:: ── CMake from Visual Studio 2022 Build Tools ─────────────────────────────
set "CMAKE_EXE=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
if not exist "%CMAKE_EXE%" (
    echo ERROR: cmake.exe not found at expected VS2022 BuildTools location.
    echo        Install "C++ CMake tools for Windows" via the VS Installer.
    exit /b 1
)

:: ── conda p2dp environment paths ──────────────────────────────────────────
set "CONDA_ENV=C:\Users\docto\miniconda3\envs\p2dp"
set "PYTHON_EXE=%CONDA_ENV%\python.exe"
set "PYBIND11_DIR=%CONDA_ENV%\lib\site-packages\pybind11\share\cmake\pybind11"
set "OpenCV_DIR=%CONDA_ENV%\Library\cmake"
set "Eigen3_DIR=%CONDA_ENV%\Library\share\eigen3\cmake"

if not exist "%PYTHON_EXE%" (
    echo ERROR: p2dp conda env not found at %CONDA_ENV%
    echo        Run:  conda create -n p2dp python=3.10
    echo              conda activate p2dp
    echo              conda install -c conda-forge opencv eigen pybind11
    exit /b 1
)

:: ── Create directories ─────────────────────────────────────────────────────
if not exist "epipolar_matching" mkdir "epipolar_matching"
set "BUILD_DIR=cpp\build"
if not exist "%BUILD_DIR%" mkdir "%BUILD_DIR%"

:: ── Configure ─────────────────────────────────────────────────────────────
echo.
echo [1/2] Configuring CMake...
"%CMAKE_EXE%" cpp -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DPYTHON_EXECUTABLE="%PYTHON_EXE%" ^
    -Dpybind11_DIR="%PYBIND11_DIR%" ^
    -DOpenCV_DIR="%OpenCV_DIR%" ^
    -DEigen3_DIR="%Eigen3_DIR%"

if errorlevel 1 (
    echo.
    echo ERROR: CMake configuration failed.
    exit /b 1
)

:: ── Build ──────────────────────────────────────────────────────────────────
echo.
echo [2/2] Building...
"%CMAKE_EXE%" --build "%BUILD_DIR%" --config Release --parallel

if errorlevel 1 (
    echo.
    echo ERROR: Build failed.
    exit /b 1
)

echo.
echo ============================================================
echo  Build succeeded.  Module is in epipolar_matching\
echo  Run the demo:
echo    conda activate p2dp
echo    python demo.py
echo ============================================================
endlocal
