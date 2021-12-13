add_library(base_edge SHARED base_edge.cpp)
cuda_add_library(base_edge_CUDA SHARED base_edge.cu)
cuda_add_library(update_CUDA SHARED update.cu)

target_link_libraries(base_edge base_edge_CUDA update_CUDA)