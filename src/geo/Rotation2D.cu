/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <geo/Geo.cuh>
#include <Wrapper.hpp>

namespace MegBA {
    namespace geo {
        namespace {
            template<typename T>
            __global__ void Rotation2DToRotation(const int nElm, const int N,
                                                 const T *R, const T *dR,
                                                 T *R00, T *R01, T *R10, T *R11,
                                                 T *dR00, T *dR01, T *dR10, T *dR11) {
                unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
                if (idx >= nElm) return;

                T r = R[idx];
                T sinr, cosr;
                Wrapper::sincosG<T>::call(r, &sinr, &cosr);
                R00[idx] = cosr;
                R01[idx] = sinr;
                R10[idx] = -sinr;
                R11[idx] = cosr;
                for (int i = 0; i < N; ++i) {
                    unsigned int index = idx + i * nElm;
                    dR00[index] *= -sinr;
                    dR01[index] *= -cosr;
                    dR10[index] *= cosr;
                    dR11[index] *= -sinr;
                }
            }
        }

        template<typename T>
        JM22 <T> Rotation2DToRotationMatrix(const Eigen::Rotation2D<JetVector<T>> &Rotation2D) {
            JM22 <T> R{};
            const auto &JV_Template = Rotation2D.angle();
            for (int i = 0; i < 2; ++i) {
                for (int j = 0; j < 2; ++j) {
                  R(i, j).InitAs(JV_Template);
                }
            }

            const auto N = JV_Template.getGradShape();
            for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
                cudaSetDevice(i);
                const auto nElm = JV_Template.get_Elm_Num(i);
                // 512 instead of 1024 for the limitation of registers
                dim3 block_dim(std::min(decltype(nElm)(768), nElm));
                dim3 grid_dim((nElm - 1) / block_dim.x + 1);
                Rotation2DToRotation<T><<<grid_dim, block_dim>>>(
                        nElm, N,
                        Rotation2D.angle().get_CUDA_Res_ptr()[i], Rotation2D.angle().get_CUDA_Grad_ptr()[i],
                        R(0, 0).get_CUDA_Res_ptr()[i], R(1, 0).get_CUDA_Res_ptr()[i],
                        R(0, 1).get_CUDA_Res_ptr()[i], R(1, 1).get_CUDA_Res_ptr()[i],
                        R(0, 0).get_CUDA_Grad_ptr()[i], R(1, 0).get_CUDA_Grad_ptr()[i],
                        R(0, 1).get_CUDA_Grad_ptr()[i], R(1, 1).get_CUDA_Grad_ptr()[i]);
            }

            // TODO: use stream sync later
            cudaDeviceSynchronize();
            return R;
        }

        template JM22<float> Rotation2DToRotationMatrix(const Eigen::Rotation2D<JetVector<float>> &Rotation2D);

        template JM22<double> Rotation2DToRotationMatrix(const Eigen::Rotation2D<JetVector<double>> &Rotation2D);
    }
}