add_library(schur_pcg_solver SHARED schur_pcg_solver.cpp)
cuda_add_library(schur_pcg_solver_CUDA SHARED schur_pcg_solver.cu)
target_link_libraries(schur_pcg_solver schur_pcg_solver_CUDA)
target_link_libraries(schur_pcg_solver_CUDA ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})
if (DEFINED MEGBA_ENABLE_NCCL)
    target_link_libraries(schur_pcg_solver_CUDA nccl)
endif ()

