/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "Common.h"
#include "cuda_runtime.h"
namespace MegBA {
    namespace math {
        namespace function {
            template<typename T>
            void Vector_add_Vector_CUDA(const JetVector<T> &f, const JetVector<T> &g, JetVector<T> &out);

            template<typename T>
            void Vector_minus_Vector_CUDA(const JetVector<T> &f, const JetVector<T> &g,
                                          JetVector<T> &out);

            template<typename T>
            void Vector_multiplies_Vector_CUDA(const JetVector<T> &f, const JetVector<T> &g,
                                               JetVector<T> &out);

            template<typename T>
            void Vector_divides_Vector_CUDA(const JetVector<T> &f, const JetVector<T> &g,
                                            JetVector<T> &out);

            template<typename T>
            void JetVector_add_Scalar_CUDA(const JetVector<T> &f, T g,
                                            JetVector<T> &out);

            template<typename T>
            void JetVector_minus_Scalar_CUDA(const JetVector<T> &f, T g,
                                              JetVector<T> &out);

            template<typename T>
            void JetVector_multiplies_Scalar_CUDA(const JetVector<T> &f, T g,
                                                   JetVector<T> &out);

            template<typename T>
            void Scalar_minus_JetVector_CUDA(T f, const JetVector<T> &g,
                                              JetVector<T> &out);

            template<typename T>
            void Scalar_divides_JetVector_CUDA(T f, const JetVector<T> &g,
                                                JetVector<T> &out);

            template<typename T>
            void abs_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out);

            template<typename T>
            void cos_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out);

            template<typename T>
            void sin_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                     MegBA::JetVector<T> &out);

            template<typename T>
            void sqrt_JetVector_CUDA(const MegBA::JetVector<T> &f,
                                      MegBA::JetVector<T> &out);
        }
    }
}
