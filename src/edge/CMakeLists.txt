add_library(edge)
add_library(MegBA::edge ALIAS edge)

target_sources(edge 
               PRIVATE 
               base_edge.cpp
               base_edge.cu
               update.cu
               build_linear_system.cu)

target_include_directories(edge 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(edge
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX
                      MegBA::operator
                      )

target_compile_options(edge PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

install(TARGETS edge
        EXPORT "${EDGE_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${EDGE_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")