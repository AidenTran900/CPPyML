# Python bindings using pybind11

cmake_minimum_required(VERSION 3.16)

# Find Python and pybind11
find_package(Python COMPONENTS Interpreter Development REQUIRED)

# Use local pybind11 clone
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../pybind11 pybind11_build)

# Source files for the ML library
set(ML_SOURCES
    ../source/math/matrix.cpp
    ../source/core/loss.cpp
    ../source/core/optimizer.cpp
    ../source/core/regularizer.cpp
    ../source/core/metrics.cpp
    ../source/models/linear-regression.cpp
    ../source/models/logistic-regression.cpp
    ../source/models/support-vector-machine.cpp
    ../source/models/k-means-clustering.cpp
    ../source/models/k-nearest-neighbors.cpp
    ../source/models/descision-tree.cpp
    ../source/models/random-forest.cpp
)

# Create Python module
pybind11_add_module(ml_lib bindings.cpp ${ML_SOURCES})

# Include directories
target_include_directories(ml_lib
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../include
        ${CMAKE_CURRENT_SOURCE_DIR}/../source
)

# Set C++ standard (C++17 required for std::clamp)
target_compile_features(ml_lib PRIVATE cxx_std_17)

# Installation
install(TARGETS ml_lib
    LIBRARY DESTINATION .
)
