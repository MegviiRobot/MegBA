/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "geo/geo.cuh"
#include "vertex/SE2_vertex.h"

namespace MegBA {
namespace {
template <typename T>
__device__ T normalize_rotation2D_dfn(T r) {
  if (r > -M_PI && r < M_PI) return r;
  r = r - floor(r / (2 * M_PI)) * 2 * M_PI;
  if (r >= M_PI) r -= 2 * M_PI;
  if (r < -M_PI) r += 2 * M_PI;
  return r;
}

template <typename T>
__global__ void normalize_rotation2D_Kernel(const int nItem, const int N,
                                            const T* R, const T* dR, T* R_out,
                                            T* dR_out) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;

  T r = -R[idx];
  R_out[idx] = r = normalize_rotation2D_dfn(r);
  int flip_r = -M_PI_2 < r && r < M_PI_2 ? -1 : 1;
  for (int i = 0; i < N; ++i) {
    unsigned int index = idx + i * nItem;
    dR_out[index] = dR[index] * flip_r;
  }
}
}  // namespace

template <typename T>
SE2<T> inverse_SE2_CUDA(const SE2<T>& se2) {
  SE2<T> se2_inv{};
  const auto& init_template = se2.rotation().angle();
  se2_inv.rotation().angle().initAs(init_template);
  se2_inv.translation()(0).initAs(init_template);
  se2_inv.translation()(1).initAs(init_template);

  const auto N = init_template.getGradShape();

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = init_template.getItemNum(i);
    dim3 block_dim(std::min(decltype(nItem)(512), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);

    normalize_rotation2D_Kernel<T><<<grid_dim, block_dim>>>(
        nItem, N, se2.rotation().angle().getCUDAResPtr()[i],
        se2.rotation().angle().getCUDAGradPtr()[i],
        se2_inv.rotation().angle().getCUDAResPtr()[i],
        se2_inv.rotation().angle().getCUDAGradPtr()[i]);
  }

  auto R = geo::Rotation2DToRotationMatrix(se2_inv.rotation());

  se2_inv.translation() = R * (-1 * se2.translation());
  return se2_inv;
}

template SE2<float> inverse_SE2_CUDA(const SE2<float>& se2);

template SE2<double> inverse_SE2_CUDA(const SE2<double>& se2);
}  // namespace MegBA