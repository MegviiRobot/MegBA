add_library(base_solver SHARED base_solver.cpp)

add_library(schur_pcg_solver SHARED schur_pcg_solver.cpp)
cuda_add_library(schur_pcg_solver_CUDA SHARED schur_pcg_solver.cu)
target_link_libraries(schur_pcg_solver schur_pcg_solver_CUDA base_solver)
target_link_libraries(schur_pcg_solver_CUDA ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} nccl)
