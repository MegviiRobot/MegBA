add_executable(BAL_Double BAL_Double.cpp)
target_link_libraries(BAL_Double base_problem base_edge_cuda JetVector cuGeo)

add_executable(BAL_Float BAL_Float.cpp)
target_link_libraries(BAL_Float base_problem base_edge_cuda JetVector cuGeo)
