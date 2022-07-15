add_library(solver)
add_library(MegBA::solver ALIAS solver)

target_sources(solver 
               PRIVATE 
               schur_pcg_solver.cpp
               schur_pcg_solver.cu
               implicit_schur_pcg_solver.cpp
               implicit_schur_pcg_solver.cu)

target_include_directories(solver 
                           PUBLIC 
                           $<BUILD_INTERFACE:${generated_headers}/>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/../../include>)

target_link_libraries(solver 
                      PRIVATE
                      Eigen3::Eigen
                      CUDA::cusparse
                      Thrust
                      Threads::Threads
                      OpenMP::OpenMP_CXX
                      MegBA::operator)

target_compile_options(solver PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr --extended-lambda>)

if(MEGBA_ENABLE_NCCL)
    target_compile_definitions(solver PRIVATE MEGBA_ENABLE_NCCL)
    target_link_libraries(solver PRIVATE nccl)
endif()

install(TARGETS solver
        EXPORT "${SOLVER_EXPORT_NAME}"
        LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
        RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
        INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")

install(EXPORT "${SOLVER_EXPORT_NAME}"
        NAMESPACE "MegBA::"
        DESTINATION "${config_install_dir}")