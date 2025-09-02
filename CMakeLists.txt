cmake_minimum_required(VERSION 3.23)
project(fbm_core_proj LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)
add_compile_options($<$<CONFIG:Release>:-O3>)
add_compile_definitions($<$<CONFIG:Release>:NDEBUG>)
option(BUILD_SHARED_LIBS "Build shared instead of static" ON)
option(USE_CBLAS "Link a system BLAS if available" OFF)
option(FBM_USE_OPENMP "Enable OpenMP if available" OFF)

# Library
add_library(fbm_core
        src/core/Simulator.cpp
        src/core/GBM_Euler.cpp
        src/core/BrownianNoise.cpp
)

target_include_directories(fbm_core PUBLIC ${CMAKE_SOURCE_DIR}/include)

# Optional OpenMP
if(FBM_USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(fbm_core PUBLIC OpenMP::OpenMP_CXX)
        target_compile_definitions(fbm_core PUBLIC FBM_USE_OPENMP)
    endif()
endif()

# Optional BLAS (future use; safe when OFF)
if(USE_CBLAS)
    if(APPLE)
        find_library(ACCELERATE Accelerate)
        if(ACCELERATE)
            target_link_libraries(fbm_core PUBLIC ${ACCELERATE})
            target_compile_definitions(fbm_core PUBLIC FBM_USE_ACCELERATE)
        endif()
    else()
        find_package(BLAS)
        if(BLAS_FOUND)
            target_link_libraries(fbm_core PUBLIC BLAS::BLAS)
            target_compile_definitions(fbm_core PUBLIC FBM_USE_BLAS)
        endif()
    endif()
endif()

# CLI
add_executable(fbm_cli src/cli/main.cpp)
target_link_libraries(fbm_cli PRIVATE fbm_core)
target_include_directories(fbm_cli PRIVATE ${CMAKE_SOURCE_DIR}/include)

# Install
install(TARGETS fbm_core fbm_cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)