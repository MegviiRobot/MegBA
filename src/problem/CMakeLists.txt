add_library(problem)
add_library(MegBA::problem ALIAS problem)

target_sources(problem 
               PRIVATE 
               base_problem.cpp
               base_problem.cu)

target_include_directories(problem 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(problem 
                      PRIVATE 
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX
                      MegBA::resource)

target_compile_options(problem PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

install(TARGETS problem
        EXPORT "${PROBLEM_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${PROBLEM_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")