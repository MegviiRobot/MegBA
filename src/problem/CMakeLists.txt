add_library(Base_Problem SHARED BaseProblem.cpp)

cuda_add_library(cuBase_Problem SHARED Base_Problem.cu)
target_link_libraries(cuBase_Problem ${CUDA_cusparse_LIBRARY})

cuda_add_library(cusolve_Linear SHARED solve_Linear.cu)
target_link_libraries(cusolve_Linear ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} nccl)

cuda_add_library(cumake_H_and_g SHARED make_H_and_g.cu)
target_link_libraries(cumake_H_and_g cuBase_Problem cusolve_Linear ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})

cuda_add_library(cuupdate SHARED update.cu)

target_link_libraries(Base_Problem cuBase_Problem cumake_H_and_g cusolve_Linear cuupdate cuManager)