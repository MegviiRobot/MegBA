# CMake 3.15+ for Thrust/Cub support
cmake_minimum_required(VERSION VERSION 3.15 FATAL_ERROR)
project(MegBA LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)

#find_package(OpenMP)

# CUDA
find_package(CUDA QUIET)
IF(CUDA_FOUND)
    message(STATUS "CUDA found")
    set(CMAKE_CUDA_RUNTIME_LIBRARY SHARED)
    enable_language(CUDA)
    set(CUDA_64_BIT_DEVICE_CODE ON FORCE)
    include_directories("${CUDA_INCLUDE_DIRS}")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")
    set(CMAKE_CUDA_ARCHITECTURES "60")
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
    if (${CMAKE_CUDA_COMPILER_VERSION} GREATER_EQUAL 11.5)
        set(Thrust_DIR /usr/local/cuda/targets/x86_64-linux/lib/cmake/thrust/)
        set(CUB_DIR /usr/local/cuda/targets/x86_64-linux/lib/cmake/cub/)
    else ()
        set(Thrust_DIR /usr/local/cuda/include/thrust/cmake/)
        set(CUB_DIR /usr/local/cuda/include/cub/cmake/)
    endif ()


ELSE()
    message("CUDA not found!")
ENDIF()

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUG=1)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0 -fPIC -fopenmp -pthread")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O0")
else()
    add_definitions(-DDEBUG=0)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -fopenmp -pthread")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3")
endif()

# Thrust
find_package(Thrust REQUIRED CONFIG)
thrust_create_target(Thrust)

include_directories(include)
include_directories(thirdparty)
add_subdirectory(src)
add_subdirectory(examples)
