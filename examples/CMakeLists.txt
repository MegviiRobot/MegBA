find_package(gflags REQUIRED)
add_executable(BAL_Double BAL_Double.cpp)
target_link_libraries(BAL_Double
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)


add_executable(BAL_Double_analytical BAL_Double_analytical.cpp)
target_link_libraries(BAL_Double_analytical
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)


add_executable(BAL_Double_analytical BAL_Double_analytical.cpp)
target_link_libraries(BAL_Double_analytical
                      PRIVATE
                      CUDA::cusparse
                      MegBA::algo
                      MegBA::resource
                      MegBA::linear_system
                      MegBA::operator
                      MegBA::problem
                      MegBA::geo
                      MegBA::edge
                      MegBA::solver
                      Eigen3::Eigen
                      gflags)

add_executable(BAL_Float BAL_Float.cpp)
target_link_libraries(BAL_Float
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)

add_executable(BAL_Float_analytical BAL_Float_analytical.cpp)
target_link_libraries(BAL_Float_analytical
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)

add_executable(BAL_Double_implicit BAL_Double_implicit.cpp)
target_link_libraries(BAL_Double_implicit
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)

add_executable(BAL_Double_analytical_implicit BAL_Double_analytical_implicit.cpp)
target_link_libraries(BAL_Double_analytical_implicit
        PRIVATE
        CUDA::cusparse
        MegBA::algo
        MegBA::resource
        MegBA::linear_system
        MegBA::operator
        MegBA::problem
        MegBA::geo
        MegBA::edge
        MegBA::solver
        Eigen3::Eigen
        gflags)