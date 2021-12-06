cuda_add_library(cuEdge_Vector SHARED Edge_Vector.cu)

add_library(Base_Edge SHARED BaseEdge.cpp)
target_link_libraries(Base_Edge cuEdge_Vector)