cuda_add_library(resource SHARED handle_manager.cpp memory_pool.cu)
target_link_libraries(resource ${CUDA_cusparse_LIBRARY} ${CUDA_cublas_LIBRARY})