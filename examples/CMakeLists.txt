add_executable(BAL_Double BAL_Double.cpp)
target_link_libraries(BAL_Double base_problem base_edge jet_vector geo)

add_executable(BAL_Float BAL_Float.cpp)
target_link_libraries(BAL_Float base_problem base_edge jet_vector geo)

add_executable(1dsfm_Double 1dsfm_Double.cpp)
target_link_libraries(1dsfm_Double base_problem base_edge jet_vector geo)
