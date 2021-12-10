add_library(base_problem SHARED base_problem.cpp)

cuda_add_library(base_problem_CUDA SHARED base_problem.cu)
target_link_libraries(base_problem_CUDA ${CUDA_cusparse_LIBRARY})

cuda_add_library(cusolve_Linear SHARED solve_Linear.cu)
target_link_libraries(cusolve_Linear ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} nccl)

cuda_add_library(cumake_H_and_g SHARED make_H_and_g.cu)
target_link_libraries(cumake_H_and_g base_problem_CUDA cusolve_Linear ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})

cuda_add_library(cuupdate SHARED update.cu)

target_link_libraries(base_problem base_problem_CUDA cumake_H_and_g cusolve_Linear cuupdate resource)