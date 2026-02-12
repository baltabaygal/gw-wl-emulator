cmake_minimum_required(VERSION 3.20)
project(gw_wl_emulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GSL REQUIRED)

# ===============================
# Core physics library
# ===============================

add_library(gwcore
    basics.cpp
    cosmology.cpp
    lensing.cpp
    lnmu_wrapper.cpp
)

target_link_libraries(gwcore
    PUBLIC GSL::gsl GSL::gslcblas m
)

target_include_directories(gwcore
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}
)

# ===============================
# Executables
# ===============================

add_executable(main_lensing
    main_lensing.cpp
)

target_link_libraries(main_lensing
    PRIVATE gwcore
)

add_executable(main_sample_lnmu
    main_sample_lnmu.cpp
)

target_link_libraries(main_sample_lnmu
    PRIVATE gwcore
)

add_executable(test_stats test_stats.cpp)
target_link_libraries(test_stats PRIVATE gwcore)

# --- Python bindings (pybind11) ---
option(BUILD_PYTHON "Build Python module gwlensing" ON)

if (BUILD_PYTHON)
  find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

  # Prefer vendored pybind11? If not vendored, use system package:
  find_package(pybind11 CONFIG REQUIRED)

  pybind11_add_module(gwlensing cpp/python_bindings.cpp)
  target_link_libraries(gwlensing PRIVATE gwcore)
  target_include_directories(gwlensing PRIVATE cpp)
endif()
