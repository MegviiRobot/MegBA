add_library(JetVector SHARED jet_vector.cpp)

cuda_add_library(cuJetVector SHARED JetVector.cu)

cuda_add_library(math_function_Jet_Vector_CUDA SHARED math_function_Jet_Vector_CUDA.cu)

add_library(math_function_Jet_Vector_CPU SHARED math_function_Jet_Vector_CPU.cpp)

target_link_libraries(JetVector
        cuJetVector
        math_function_Jet_Vector_CPU
        math_function_Jet_Vector_CUDA)
