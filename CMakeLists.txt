cmake_minimum_required(VERSION 3.20)
project(RoughVolatility-ShortTimeApproximation
  VERSION 0.1.0
  LANGUAGES CXX)

# Options
option(FBM_BUILD_TESTS "Build test suite" ON)
option(FBM_USE_OPENMP "Enable OpenMP parallelism" OFF)
option(FBM_USE_BLAS   "Enable BLAS dgemm path" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMP (optional)
if(FBM_USE_OPENMP)
  find_package(OpenMP REQUIRED)
endif()

# BLAS (optional) – define only (linking left to user/toolchain)
if(FBM_USE_BLAS)
  add_compile_definitions(FBM_USE_BLAS)
endif()

add_library(fbm_core
  src/core/Simulator.cpp
  src/core/GBMEuler.cpp
  src/core/BrownianNoise.cpp
  src/core/VolterraKernelBrownian.cpp
  src/core/VolterraKernelPowerLaw.cpp
  src/core/VolterraNoise.cpp
  src/core/RoughBergomiFactor.cpp
  src/core/RoughBergomiAssetEuler.cpp
)

target_include_directories(fbm_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

if(FBM_USE_OPENMP)
  target_link_libraries(fbm_core PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(fbm_cli
  src/cli/main.cpp
)

target_link_libraries(fbm_cli PRIVATE fbm_core)

# Tests
if(FBM_BUILD_TESTS)
  add_executable(fbm_tests
    tests/test_antithetic.cpp
    tests/test_corr.cpp
    tests/test_bs.cpp
    tests/test_noise.cpp
    tests/test_kernel.cpp
    tests/test_kernel_powerlaw.cpp
    tests/test_rb_factor.cpp
    tests/test_rb_asset.cpp
    tests/test_rb_curve.cpp
    tests/test_cli_alignment.cpp
    tests/test_kernel_grid.cpp
    third_party/catch2/catch_amalgamated.cpp
  )
  target_link_libraries(fbm_tests PRIVATE fbm_core)
endif()

# Install
include(GNUInstallDirs)
install(TARGETS fbm_core fbm_cli
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

# Export minimal package config (optional lightweight)
install(EXPORT fbmTargets
  FILE fbmTargets.cmake
  NAMESPACE fbm::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fbm)

include(CMakePackageConfigHelpers)
configure_package_config_file(
  ${CMAKE_CURRENT_LIST_DIR}/cmake/fbmConfig.cmake.in
  ${CMAKE_CURRENT_BINARY_DIR}/fbmConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fbm
)
write_basic_package_version_file(
  ${CMAKE_CURRENT_BINARY_DIR}/fbmConfigVersion.cmake
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY SameMajorVersion
)
install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/fbmConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/fbmConfigVersion.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/fbm)
