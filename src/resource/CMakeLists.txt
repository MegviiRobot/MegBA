cuda_add_library(cuManager SHARED Manager.cpp MemoryPool.cu)
target_link_libraries(cuManager ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY} ${CUDA_cusolver_LIBRARY})