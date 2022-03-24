/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "geo/geo.cuh"
#include "wrapper.hpp"

namespace MegBA {
namespace geo {
namespace {
template <typename T>
__global__ void Rotation2DToRotation(const int nItem, const int N, const T *R,
                                     const T *dR, T *R00, T *R01, T *R10,
                                     T *R11, T *dR00, T *dR01, T *dR10,
                                     T *dR11) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;

  T r = R[idx];
  T sinr, cosr;
  Wrapper::sincosG<T>::call(r, &sinr, &cosr);
  R00[idx] = cosr;
  R01[idx] = sinr;
  R10[idx] = -sinr;
  R11[idx] = cosr;
  for (int i = 0; i < N; ++i) {
    unsigned int index = idx + i * nItem;
    dR00[index] *= -sinr;
    dR01[index] *= -cosr;
    dR10[index] *= cosr;
    dR11[index] *= -sinr;
  }
}
}  // namespace

template <typename T>
JM22<T> Rotation2DToRotationMatrix(
    const Eigen::Rotation2D<JetVector<T>> &Rotation2D) {
  JM22<T> R{};
  const auto &JV_Template = Rotation2D.angle();
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 2; ++j) {
      R(i, j).initAs(JV_Template);
    }
  }

  const auto N = JV_Template.getGradShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = JV_Template.getItemNum(i);
    // 512 instead of 1024 for the limitation of registers
    dim3 block_dim(std::min(decltype(nItem)(768), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    Rotation2DToRotation<T><<<grid_dim, block_dim>>>(
        nItem, N, Rotation2D.angle().getCUDAResPtr()[i],
        Rotation2D.angle().getCUDAGradPtr()[i], R(0, 0).getCUDAResPtr()[i],
        R(1, 0).getCUDAResPtr()[i], R(0, 1).getCUDAResPtr()[i],
        R(1, 1).getCUDAResPtr()[i], R(0, 0).getCUDAGradPtr()[i],
        R(1, 0).getCUDAGradPtr()[i], R(0, 1).getCUDAGradPtr()[i],
        R(1, 1).getCUDAGradPtr()[i]);
  }

  // TODO: use stream sync later
  cudaDeviceSynchronize();
  return R;
}

template JM22<float> Rotation2DToRotationMatrix(
    const Eigen::Rotation2D<JetVector<float>> &Rotation2D);

template JM22<double> Rotation2DToRotationMatrix(
    const Eigen::Rotation2D<JetVector<double>> &Rotation2D);
}  // namespace geo
}  // namespace MegBA