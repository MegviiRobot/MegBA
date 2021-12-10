cuda_add_library(base_edge_CUDA SHARED base_edge.cu)

add_library(base_edge_cuda SHARED base_edge.cpp)
target_link_libraries(base_edge_cuda base_edge_CUDA)