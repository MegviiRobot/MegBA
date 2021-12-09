/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <geo/Geo.cuh>

namespace MegBA {
namespace geo {
namespace {
template <typename T>
__global__ void RadialDistortionNoGradKernel(
    const int nElm, const int N, const T *px_da_ptr, const T *py_da_ptr,
    const T *px_dv_ptr, const T *py_dv_ptr, const T *f_ptr, const T *k1_ptr,
    const T *k2_ptr, T *da_ptr, T *dv_ptr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nElm)
    return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_da_ptr[tid];

  T py = py_da_ptr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i)
    dv_ptr[tid + nElm * i] = partial * (px_dv_ptr[tid + nElm * i] * px +
                                        py_dv_ptr[tid + nElm * i] * py);

  da_ptr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
}

template <typename T>
__global__ void
RadialDistortionKernel(const int nElm, const int N, const T *px_da_ptr,
                       const T *py_da_ptr, const T *px_dv_ptr,
                       const T *py_dv_ptr, const T *f_ptr, const T *k1_ptr,
                       const T *k2_ptr, const T *f_dv_ptr, const T *k1_dv_ptr,
                       const T *k2_dv_ptr, T *da_ptr, T *dv_ptr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nElm)
    return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_da_ptr[tid];

  T py = py_da_ptr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i) {
    unsigned int index = tid + nElm * i;
    dv_ptr[index] =
        partial *
            (px_dv_ptr[tid + nElm * i] * px + py_dv_ptr[tid + nElm * i] * py) +
        f_dv_ptr[index] * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2) +
        k1_dv_ptr[index] * f * l2_pow2 +
        k2_dv_ptr[index] * f * l2_pow2 * l2_pow2;
  }

  da_ptr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
}

template <typename T>
__global__ void RadialDistortionFastGradKernel(
    const int nElm, const int N, const T *px_da_ptr, const T *py_da_ptr,
    const T *px_dv_ptr, const T *py_dv_ptr, const T *f_ptr, const T *k1_ptr,
    const T *k2_ptr, const int f_grad_position, const int k1_grad_position,
    const int k2_grad_position, T *da_ptr, T *dv_ptr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nElm)
    return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_da_ptr[tid];

  T py = py_da_ptr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i) {
    unsigned int index = tid + nElm * i;
    dv_ptr[index] = partial * (px_dv_ptr[tid + nElm * i] * px +
                               py_dv_ptr[tid + nElm * i] * py) +
                    (i == f_grad_position ? 1 : 0) *
                        (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2) +
                    (i == k1_grad_position ? 1 : 0) * f * l2_pow2 +
                    (i == k2_grad_position ? 1 : 0) * f * l2_pow2 * l2_pow2;
  }

  da_ptr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
}

template <typename T>
void RadialDistortionImpl(const JV3<T> &point, const JV3<T> &intrinsic,
                          JetVector<T> *out) {
  const auto N = out->getGradShape();
  bool use_fast_grad{true};
  for (int i = 0; i < 3; ++i)
    use_fast_grad &= intrinsic(i).getGradPosition() != -1;

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = out->getElmNum(i);
    dim3 block_dim(std::min(decltype(nElm)(256), nElm));
    dim3 grid_dim((nElm - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nElm, N, point(0).getCUDAResPtr()[i],
          point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
          point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
          intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getGradPosition(), intrinsic(1).getGradPosition(),
            intrinsic(2).getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i],
            intrinsic(2).getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
void RadialDistortionImpl(const JV3<T> &point,
                          const Eigen::Map<const JV3<T>> &intrinsic,
                          JetVector<T> *out) {
  const auto N = out->getGradShape();
  bool use_fast_grad{true};
  for (int i = 0; i < 3; ++i)
    use_fast_grad &= intrinsic(i).getGradPosition() != -1;

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = out->getElmNum(i);
    dim3 block_dim(std::min(decltype(nElm)(256), nElm));
    dim3 grid_dim((nElm - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nElm, N, point(0).getCUDAResPtr()[i],
          point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
          point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
          intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getGradPosition(), intrinsic(1).getGradPosition(),
            intrinsic(2).getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i],
            intrinsic(2).getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      }
    }
  }
}

template <typename T>
void RadialDistortionImpl(const JV3<T> &point,
                          const Eigen::Map<const JVD<T>> &intrinsic,
                          JetVector<T> *out) {
  const auto N = out->getGradShape();
  bool use_fast_grad{true};
  for (int i = 0; i < 3; ++i)
    use_fast_grad &= intrinsic(i).getGradPosition() != -1;

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = out->getElmNum(i);
    dim3 block_dim(std::min(decltype(nElm)(256), nElm));
    dim3 grid_dim((nElm - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nElm, N, point(0).getCUDAResPtr()[i],
          point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
          point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
          intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getGradPosition(), intrinsic(1).getGradPosition(),
            intrinsic(2).getGradPosition(), out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nElm, N, point(0).getCUDAResPtr()[i],
            point(1).getCUDAResPtr()[i], point(0).getCUDAGradPtr()[i],
            point(1).getCUDAGradPtr()[i], intrinsic(0).getCUDAResPtr()[i],
            intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i],
            intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i],
            intrinsic(2).getCUDAGradPtr()[i], out->getCUDAResPtr()[i],
            out->getCUDAGradPtr()[i]);
      }
    }
  }
}
}

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point, const JV3<T> &intrinsic) {
  return JetVector<T>{point(0, 0), [&](JetVector<T> *out) {
                         RadialDistortionImpl(point, intrinsic, out);
                       }};
}

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point,
                               const Eigen::Map<const JV3<T>> &intrinsic) {
  return JetVector<T>{point(0), [&](JetVector<T> *out) {
                         RadialDistortionImpl(point, intrinsic, out);
                       }};
}

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point,
                               const Eigen::Map<const JVD<T>> &intrinsic) {
  assert(intrinsic.rows() == 3 && intrinsic.cols() == 1);
  return JetVector<T>{point(0), [&](JetVector<T> *out) {
                         RadialDistortionImpl(point, intrinsic, out);
                       }};
}

template JetVector<float> RadialDistortion(const JV3<float> &point,
                                            const JV3<float> &intrinsic);
template JetVector<double> RadialDistortion(const JV3<double> &point,
                                             const JV3<double> &intrinsic);
template JetVector<float>
RadialDistortion(const JV3<float> &point,
                 const Eigen::Map<const JV3<float>> &intrinsic);
template JetVector<double>
RadialDistortion(const JV3<double> &point,
                 const Eigen::Map<const JV3<double>> &intrinsic);
template JetVector<float>
RadialDistortion(const JV3<float> &point,
                 const Eigen::Map<const JVD<float>> &intrinsic);
template JetVector<double>
RadialDistortion(const JV3<double> &point,
                 const Eigen::Map<const JVD<double>> &intrinsic);
}
}