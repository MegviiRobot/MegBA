cuda_add_library(cuManager SHARED Manager.cpp Memory_Pool.cu)
target_link_libraries(cuManager ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})