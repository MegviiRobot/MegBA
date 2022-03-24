/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "geo/geo.cuh"

namespace MegBA {
namespace geo {
namespace {
template <typename T>
__global__ void RadialDistortionNoGradKernel(
    const int nItem, const int N, const T *px_valueDevicePtr,
    const T *py_valueDevicePtr, const T *px_gradDevicePtr,
    const T *py_gradDevicePtr, const T *f_ptr, const T *k1_ptr, const T *k2_ptr,
    T *valueDevicePtr, T *gradDevicePtr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_valueDevicePtr[tid];

  T py = py_valueDevicePtr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i)
    gradDevicePtr[tid + nItem * i] =
        partial * (px_gradDevicePtr[tid + nItem * i] * px +
                   py_gradDevicePtr[tid + nItem * i] * py);

  valueDevicePtr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
}

template <typename T>
__global__ void RadialDistortionKernel(
    const int nItem, const int N, const T *px_valueDevicePtr,
    const T *py_valueDevicePtr, const T *px_gradDevicePtr,
    const T *py_gradDevicePtr, const T *f_ptr, const T *k1_ptr, const T *k2_ptr,
    const T *f_gradDevicePtr, const T *k1_gradDevicePtr,
    const T *k2_gradDevicePtr, T *valueDevicePtr, T *gradDevicePtr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_valueDevicePtr[tid];

  T py = py_valueDevicePtr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i) {
    unsigned int index = tid + nItem * i;
    gradDevicePtr[index] = partial * (px_gradDevicePtr[tid + nItem * i] * px +
                                      py_gradDevicePtr[tid + nItem * i] * py) +
                           f_gradDevicePtr[index] *
                               (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2) +
                           k1_gradDevicePtr[index] * f * l2_pow2 +
                           k2_gradDevicePtr[index] * f * l2_pow2 * l2_pow2;
  }

  valueDevicePtr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
}

template <typename T>
__global__ void RadialDistortionFastGradKernel(
    const int nItem, const int N, const T *px_valueDevicePtr,
    const T *py_valueDevicePtr, const T *px_gradDevicePtr,
    const T *py_gradDevicePtr, const T *f_ptr, const T *k1_ptr, const T *k2_ptr,
    const int f_grad_position, const int k1_grad_position,
    const int k2_grad_position, T *valueDevicePtr, T *gradDevicePtr) {
  unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= nItem) return;
  T f = f_ptr[tid], k1 = k1_ptr[tid], k2 = k2_ptr[tid];

  T px = px_valueDevicePtr[tid];

  T py = py_valueDevicePtr[tid];

  T l2_pow2 = px * px + py * py;

  T partial = 2 * f * (k1 + 2 * k2 * l2_pow2);
  for (unsigned int i = 0; i < N; ++i) {
    unsigned int index = tid + nItem * i;
    gradDevicePtr[index] =
        partial * (px_gradDevicePtr[tid + nItem * i] * px +
                   py_gradDevicePtr[tid + nItem * i] * py) +
        (i == f_grad_position ? 1 : 0) *
            (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2) +
        (i == k1_grad_position ? 1 : 0) * f * l2_pow2 +
        (i == k2_grad_position ? 1 : 0) * f * l2_pow2 * l2_pow2;
  }

  valueDevicePtr[tid] = f * (T(1.) + k1 * l2_pow2 + k2 * l2_pow2 * l2_pow2);
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
    const auto nItem = out->getItemNum(i);
    dim3 block_dim(std::min(decltype(nItem)(256), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
          point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
          intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getGradPosition(),
            intrinsic(1).getGradPosition(), intrinsic(2).getGradPosition(),
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i], intrinsic(2).getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
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
    const auto nItem = out->getItemNum(i);
    dim3 block_dim(std::min(decltype(nItem)(256), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
          point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
          intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getGradPosition(),
            intrinsic(1).getGradPosition(), intrinsic(2).getGradPosition(),
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i], intrinsic(2).getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
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
    const auto nItem = out->getItemNum(i);
    dim3 block_dim(std::min(decltype(nItem)(256), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    if (intrinsic(0).getGradShape() == 0) {
      RadialDistortionNoGradKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
          point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
          intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
          intrinsic(2).getCUDAResPtr()[i], out->getCUDAResPtr()[i],
          out->getCUDAGradPtr()[i]);
    } else {
      if (use_fast_grad) {
        RadialDistortionFastGradKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getGradPosition(),
            intrinsic(1).getGradPosition(), intrinsic(2).getGradPosition(),
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      } else {
        RadialDistortionKernel<T><<<grid_dim, block_dim>>>(
            nItem, N, point(0).getCUDAResPtr()[i], point(1).getCUDAResPtr()[i],
            point(0).getCUDAGradPtr()[i], point(1).getCUDAGradPtr()[i],
            intrinsic(0).getCUDAResPtr()[i], intrinsic(1).getCUDAResPtr()[i],
            intrinsic(2).getCUDAResPtr()[i], intrinsic(0).getCUDAGradPtr()[i],
            intrinsic(1).getCUDAGradPtr()[i], intrinsic(2).getCUDAGradPtr()[i],
            out->getCUDAResPtr()[i], out->getCUDAGradPtr()[i]);
      }
    }
  }
}
}  // namespace

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
template JetVector<float> RadialDistortion(
    const JV3<float> &point, const Eigen::Map<const JV3<float>> &intrinsic);
template JetVector<double> RadialDistortion(
    const JV3<double> &point, const Eigen::Map<const JV3<double>> &intrinsic);
template JetVector<float> RadialDistortion(
    const JV3<float> &point, const Eigen::Map<const JVD<float>> &intrinsic);
template JetVector<double> RadialDistortion(
    const JV3<double> &point, const Eigen::Map<const JVD<double>> &intrinsic);
}  // namespace geo
}  // namespace MegBA