cmake_minimum_required(VERSION 3.14)
project(epipolar_matching)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Eigen optimizations ────────────────────────────────────────────────────────
add_definitions(-DEIGEN_NO_DEBUG -DEIGEN_FAST_MATH)
if(MSVC)
    add_definitions(-DEIGEN_STRONG_INLINE=__forceinline)
endif()

# ── Architecture check (Windows) ──────────────────────────────────────────────
if(WIN32 AND MSVC)
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        message(STATUS "Building for 64-bit Windows")
    else()
        message(WARNING "Building for 32-bit – this may cause issues with 64-bit Python")
    endif()
endif()

# ── Dependencies ──────────────────────────────────────────────────────────────
find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# ── Compiler flags ────────────────────────────────────────────────────────────
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob2 /DNDEBUG /GL /arch:AVX2 /fp:fast /DEIGEN_NO_DEBUG")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE    "${CMAKE_EXE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_MODULE_LINKER_FLAGS_RELEASE "${CMAKE_MODULE_LINKER_FLAGS_RELEASE} /LTCG")
    set(CMAKE_SHARED_LINKER_FLAGS_RELEASE "${CMAKE_SHARED_LINKER_FLAGS_RELEASE} /LTCG")
elseif(APPLE)
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -ffast-math -flto -DEIGEN_NO_DEBUG -fPIC -DNDEBUG")
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fopenmp")
    endif()
else()
    set(CMAKE_CXX_FLAGS_RELEASE
        "-O3 -march=native -mtune=native -ffast-math -funsafe-math-optimizations \
         -flto -fopenmp -fopenmp-simd -DEIGEN_NO_DEBUG -fPIC -DNDEBUG")
endif()

# ── Python extension ──────────────────────────────────────────────────────────
pybind11_add_module(guided_desc_match src/guided_desc_match.cpp)

target_link_libraries(guided_desc_match PRIVATE Eigen3::Eigen)

target_include_directories(guided_desc_match PRIVATE
    ${EIGEN3_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/src
)

find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    target_link_libraries(guided_desc_match PRIVATE OpenMP::OpenMP_CXX)
else()
    message(STATUS "OpenMP not found – building without parallel support")
endif()

# ── Output directory ──────────────────────────────────────────────────────────
# Put the .pyd / .so directly into epipolar_matching/ at the repo root.
# On multi-config generators (Visual Studio) we must set all config variants.
set(_OUT_DIR "${CMAKE_SOURCE_DIR}/../epipolar_matching")

set_target_properties(guided_desc_match PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY                 "${_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_RELEASE         "${_OUT_DIR}"
    LIBRARY_OUTPUT_DIRECTORY_DEBUG           "${_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY                 "${_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_RELEASE         "${_OUT_DIR}"
    RUNTIME_OUTPUT_DIRECTORY_DEBUG           "${_OUT_DIR}"
)

install(TARGETS guided_desc_match LIBRARY DESTINATION epipolar_matching)
