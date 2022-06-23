add_library(operator)
add_library(MegBA::operator ALIAS operator)

target_sources(operator 
               PRIVATE 
               jet_vector.cpp
               jet_vector.cu
               jet_vector_math_impl.cpp
               jet_vector_math_impl.cu)

target_include_directories(operator 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(operator
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX)
target_compile_options(operator PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

install(TARGETS operator
        EXPORT "${OPERATOR_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${OPERATOR_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")