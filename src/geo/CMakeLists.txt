add_library(geo)
add_library(MegBA::geo ALIAS geo)

target_sources(geo 
               PRIVATE 
               angle_axis.cu
               distortion.cu
               rotation2D.cu
               analytical_derivatives.cu
)

target_include_directories(geo 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(geo 
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX
                      MegBA::operator)

target_compile_options(geo 
                       PRIVATE 
                       $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

install(TARGETS geo
        EXPORT "${GEO_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${GEO_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")