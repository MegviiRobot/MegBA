add_library(base_algo SHARED base_algo.cpp)
add_library(lm_algo SHARED lm_algo.cpp)
cuda_add_library(lm_algo_CUDA SHARED lm_algo.cu)
target_link_libraries(lm_algo lm_algo_CUDA base_algo)