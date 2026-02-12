cmake_minimum_required(VERSION 3.20)
project(gw_wl_emulator LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(GSL REQUIRED)

add_executable(main_lensing
    basics.cpp
    cosmology.cpp
    lensing.cpp
    main_lensing.cpp
)

target_link_libraries(main_lensing PRIVATE GSL::gsl GSL::gslcblas m)

add_executable(main_sample_lnmu
  basics.cpp
  cosmology.cpp
  lensing.cpp
  main_sample_lnmu.cpp
)
target_link_libraries(main_sample_lnmu PRIVATE GSL::gsl GSL::gslcblas m)

