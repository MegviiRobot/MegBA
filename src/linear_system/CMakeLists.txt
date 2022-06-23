add_library(linear_system)
add_library(MegBA::linear_system ALIAS linear_system)

target_sources(linear_system 
               PRIVATE 
               base_linear_system.cpp base_linear_system.cu
               schur_linear_system.cpp schur_linear_system.cu
               LM_linear_system.cpp LM_linear_system.cu
               schur_LM_linear_system.cpp schur_LM_linear_system.cu)

target_include_directories(linear_system 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(linear_system
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      CUDA::cublas
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX)

target_compile_options(linear_system PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

if(MEGBA_ENABLE_NCCL)
    target_compile_definitions(linear_system PRIVATE MEGBA_ENABLE_NCCL)
    target_link_libraries(linear_system PRIVATE nccl)
endif()

install(TARGETS linear_system
        EXPORT "${LINEAR_SYSTEM_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${LINEAR_SYSTEM_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")