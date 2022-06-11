add_library(resource)
add_library(MegBA::resource ALIAS resource)

target_sources(resource 
               PRIVATE 
               handle_manager.cpp
               memory_pool.cu)

target_include_directories(resource 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(resource 
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      CUDA::cublas
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX)

target_compile_options(resource PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

if(MEGBA_ENABLE_NCCL)
    target_compile_definitions(resource PRIVATE MEGBA_ENABLE_NCCL)
    target_link_libraries(resource PRIVATE nccl)
endif()

install(TARGETS resource
        EXPORT "${RESOURCE_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${RESOURCE_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")