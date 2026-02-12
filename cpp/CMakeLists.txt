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
