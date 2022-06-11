add_library(algo)
add_library(MegBA::algo ALIAS algo)

target_sources(algo 
               PRIVATE 
               base_algo.cpp 
               lm_algo.cpp 
               lm_algo.cu)

target_include_directories(algo 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(algo
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads)

target_compile_options(algo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

install(TARGETS algo
        EXPORT "${ALGO_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${ALGO_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")