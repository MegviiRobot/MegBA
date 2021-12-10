add_library(base_problem SHARED base_problem.cpp)
cuda_add_library(base_problem_CUDA SHARED base_problem.cu)

cuda_add_library(solve_linear SHARED solve_linear.cu)
target_link_libraries(solve_linear ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} nccl)

cuda_add_library(build_linear_system SHARED build_linear_system.cu)

cuda_add_library(cuupdate SHARED update.cu)

target_link_libraries(base_problem base_problem_CUDA build_linear_system solve_linear cuupdate resource)