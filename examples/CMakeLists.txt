add_executable(BAL_Double BAL_Double.cpp)
target_link_libraries(BAL_Double Base_Problem Base_Edge JetVector cuGeo)

add_executable(BAL_Float BAL_Float.cpp)
target_link_libraries(BAL_Float Base_Problem Base_Edge JetVector cuGeo)
