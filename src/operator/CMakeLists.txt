add_library(jet_vector SHARED jet_vector.cpp)

cuda_add_library(jet_vector_CUDA SHARED jet_vector.cu)

cuda_add_library(math_function_Jet_Vector_CUDA SHARED math_function_Jet_Vector_CUDA.cu)

add_library(math_function_Jet_Vector_CPU SHARED math_function_Jet_Vector_CPU.cpp)

target_link_libraries(jet_vector
        jet_vector_CUDA
        math_function_Jet_Vector_CPU
        math_function_Jet_Vector_CUDA)
