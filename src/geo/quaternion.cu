/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "Wrapper.hpp"
#include "geo/Geo.cuh"

namespace MegAutoBA {
namespace geo {
namespace {
template <typename T>
__global__ void QuaternionToRotation(
    const int nItem, const int N, const T *Qx, const T *Qy, const T *Qz,
    const T *Qw, const T *dQx, const T *dQy, const T *dQz, const T *dQw, T *R00,
    T *R01, T *R02, T *R10, T *R11, T *R12, T *R20, T *R21, T *R22, T *dR00,
    T *dR01, T *dR02, T *dR10, T *dR11, T *dR12, T *dR20, T *dR21, T *dR22) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;

  const T qw = Qw[idx];
  const T qx = Qx[idx];
  const T qy = Qy[idx];
  const T qz = Qz[idx];
  R00[idx] = 1 - 2 * (qy * qy + qz * qz);
  R01[idx] = 2 * (qx * qy - qw * qz);
  R02[idx] = 2 * (qx * qz + qw * qy);
  R10[idx] = 2 * (qx * qy + qw * qz);
  R11[idx] = 1 - 2 * (qx * qx + qz * qz);
  R12[idx] = 2 * (qy * qz - qw * qx);
  R20[idx] = 2 * (qx * qz - qw * qy);
  R21[idx] = 2 * (qy * qz + qw * qx);
  R22[idx] = 1 - 2 * (qx * qx + qy * qy);

  for (int i = 0; i < N; ++i) {
    unsigned int index = idx + i * nItem;
    const T dqw = dQw[index];
    const T dqx = dQx[index];
    const T dqy = dQy[index];
    const T dqz = dQz[index];

    dR00[index] = -4 * (qy * dqy + qz * dqz);
    dR01[index] = 2 * (qx * dqy + qy * dqx - qw * dqz - qz * dqw);
    dR02[index] = 2 * (qx * dqz + qz * dqx + qw * dqy + qy * dqw);
    dR10[index] = 2 * (qx * dqy + qy * dqx + qw * dqz + qz * dqw);
    dR11[index] = -4 * (qx * dqx + qz * dqz);
    dR12[index] = 2 * (qy * dqz + qz * dqy - qw * dqx - qx * dqw);
    dR20[index] = 2 * (qx * dqz + qz * dqx - qw * dqy - qy * dqw);
    dR21[index] = 2 * (qy * dqz + qz * dqy + qw * dqx + qx * dqw);
    dR22[index] = -4 * (qx * dqx + qy * dqy);
  }
}

template <typename T>
__device__ inline int Get_i_For_R2Q(const T r[3][3]) {
  int i = 0;
  if (r[1][1] > r[0][0]) i = 1;
  if (r[2][2] > r[i][i]) i = 2;
  return i;
}

template <typename T>
struct R2Q_Address_Wrapper {};

__constant__ const float *f_R[3][3];
__constant__ const float *f_dR[3][3];
__constant__ float *f_Q[4];
__constant__ float *f_dQ[4];

template <>
struct R2Q_Address_Wrapper<float> {
  static __device__ __host__ const float *const (&get_R())[3][3] { return f_R; }
  static __device__ __host__ const float *const (&get_dR())[3][3] {
    return f_dR;
  }
  static __device__ __host__ float *const (&get_Q())[4] { return f_Q; }
  static __device__ __host__ float *const (&get_dQ())[4] { return f_dQ; }
};

__constant__ const double *d_R[3][3];
__constant__ const double *d_dR[3][3];

// x, y, z, w
__constant__ double *d_Q[4];
__constant__ double *d_dQ[4];

template <>
struct R2Q_Address_Wrapper<double> {
  static __device__ __host__ const double *const (&get_R())[3][3] {
    return d_R;
  }
  static __device__ __host__ const double *const (&get_dR())[3][3] {
    return d_dR;
  }
  static __device__ __host__ double *const (&get_Q())[4] { return d_Q; }
  static __device__ __host__ double *const (&get_dQ())[4] { return d_dQ; }
};

template <typename T>
__global__ void RotationToQuaternion(const int nItem, const int N) {
  /*
   * 00: 0    01: 1   02: 2
   * 10: 3    11: 4   12: 5
   * 20: 6    21: 7   22: 8
   */
  using W = R2Q_Address_Wrapper<T>;
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;

  const T r[3][3]{
      {W::get_R()[0][0][idx], W::get_R()[0][1][idx], W::get_R()[0][2][idx]},
      {W::get_R()[1][0][idx], W::get_R()[1][1][idx], W::get_R()[1][2][idx]},
      {W::get_R()[2][0][idx], W::get_R()[2][1][idx], W::get_R()[2][2][idx]}};

  // inv_qw start with trace
  T inv_qw = r[0][0] + r[1][1] + r[2][2];
  if (inv_qw > 0) {
    // inv_qw start with 1 / sqrt(trace + 1)
    inv_qw = 2 * Wrapper::rsqrtG<T>::call(inv_qw + 1);
    // qw_mul_4 start with inv_4qw
    T qw_mul_4 = 0.25 * inv_qw;
    W::get_Q()[0][idx] = (r[1][2] - r[2][1]) * qw_mul_4;
    W::get_Q()[1][idx] = (r[2][0] - r[0][2]) * qw_mul_4;
    W::get_Q()[2][idx] = (r[0][1] - r[1][0]) * qw_mul_4;

    // qw_mul_4 should be the inv of inv_4qw
    qw_mul_4 = 1 / qw_mul_4;
    W::get_Q()[3][idx] = 0.25 * qw_mul_4;

    const T inv_4q0_pow2 = inv_qw * inv_qw * 0.0625;
    for (int i = 0; i < N; ++i) {
      unsigned int index = idx + i * nItem;
      const T dqw = 0.125 * inv_qw *
                    (W::get_dR()[0][0][index] + W::get_dR()[1][1][index] +
                     W::get_dR()[2][2][index]);

      // w
      W::get_dQ()[3][index] = dqw;
      // x
      W::get_dQ()[0][index] =
          ((W::get_dR()[1][2][index] - W::get_dR()[2][1][index]) * qw_mul_4 -
           4 * dqw * (r[1][2] - r[2][1])) *
          inv_4q0_pow2;
      // y
      W::get_dQ()[1][index] =
          ((W::get_dR()[2][0][index] - W::get_dR()[0][2][index]) * qw_mul_4 -
           4 * dqw * (r[2][0] - r[0][2])) *
          inv_4q0_pow2;
      // z
      W::get_dQ()[2][index] =
          ((W::get_dR()[0][1][index] - W::get_dR()[1][0][index]) * qw_mul_4 -
           4 * dqw * (r[0][1] - r[1][0])) *
          inv_4q0_pow2;
    }
  } else {
    const int i = Get_i_For_R2Q(r);
    const int j = (i + 1) % 3;
    const int k = (j + 1) % 3;

    // inv_qw start with 1 / sqrt(trace + 1)
    inv_qw = 2 * Wrapper::rsqrtG<T>::call(r[i][i] - r[j][j] - r[k][k] + 1);
    // qw_mul_4 start with inv_4qw
    T qw_mul_4 = 0.25 * inv_qw;

    W::get_Q()[i][idx] = 1 / inv_qw;
    // w
    W::get_Q()[3][idx] = (r[k][j] - r[j][k]) * qw_mul_4;
    W::get_Q()[j][idx] = (r[j][i] + r[i][j]) * qw_mul_4;
    W::get_Q()[k][idx] = (r[k][i] + r[i][k]) * qw_mul_4;

    // qw_mul_4 should be the inv of inv_4qw
    qw_mul_4 = 1 / qw_mul_4;

    const T inv_4q0_pow2 = inv_qw * inv_qw * 0.0625;
    for (int n = 0; n < N; ++n) {
      unsigned int index = idx + n * nItem;
      const T dq0 = 0.125 * inv_qw *
                    (W::get_dR()[i][i][index] - W::get_dR()[j][j][index] -
                     W::get_dR()[k][k][index]);

      W::get_dQ()[i][index] = dq0;
      // w
      W::get_dQ()[3][index] =
          ((W::get_dR()[k][j][index] - W::get_dR()[j][k][index]) * qw_mul_4 -
           4 * dq0 * (r[k][j] - r[j][k])) *
          inv_4q0_pow2;
      W::get_dQ()[j][index] =
          ((W::get_dR()[j][i][index] + W::get_dR()[i][j][index]) * qw_mul_4 -
           4 * dq0 * (r[j][i] + r[i][j])) *
          inv_4q0_pow2;
      W::get_dQ()[k][index] =
          ((W::get_dR()[k][i][index] + W::get_dR()[i][k][index]) * qw_mul_4 -
           4 * dq0 * (r[k][i] + r[i][k])) *
          inv_4q0_pow2;
    }
  }
}

template <typename T>
__global__ void Normalize_(const int nItem, const int N, T *Qx, T *Qy, T *Qz,
                           T *Qw, T *dQx, T *dQy, T *dQz, T *dQw) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;

  const T qw = Qw[idx];
  const T qx = Qx[idx];
  const T qy = Qy[idx];
  const T qz = Qz[idx];
  int sign = qw > 0 ? 1 : -1;

  const T inv_l2 =
      Wrapper::rsqrtG<T>::call(qw * qw + qx * qx + qy * qy + qz * qz) * sign;

  Qw[idx] = qw * inv_l2;
  Qx[idx] = qx * inv_l2;
  Qy[idx] = qy * inv_l2;
  Qz[idx] = qz * inv_l2;

  for (int i = 0; i < N; ++i) {
    unsigned int index = idx + i * nItem;
    const T dqw = dQw[index];
    const T dqx = dQx[index];
    const T dqy = dQy[index];
    const T dqz = dQz[index];
    const T common_coeff =
        inv_l2 * inv_l2 * (qw * dqw + qx * dqx + qy * dqy + qz * dqz);
    dQw[index] = inv_l2 * (dqw - qw * common_coeff);
    dQx[index] = inv_l2 * (dqx - qx * common_coeff);
    dQy[index] = inv_l2 * (dqy - qy * common_coeff);
    dQz[index] = inv_l2 * (dqz - qz * common_coeff);
  }
}
}  // namespace

template <typename T>
JM33<T> QuaternionToRotationMatrix(const JV4<T> &Q) {
  JM33<T> R{};
  const MegAutoBA::JetVector<T> &JV_Template = Q(0, 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R(i, j).initAs(JV_Template);
    }
  }

  const auto nItem = JV_Template.getItemNum();
  const auto N = JV_Template.getGradShape();
  // 512 instead of 1024 for the limitation of registers
  dim3 block_dim(std::min(decltype(nItem)(512), nItem));
  dim3 grid_dim((nItem - 1) / block_dim.x + 1);
  QuaternionToRotation<T><<<grid_dim, block_dim>>>(
      nItem, N, Q.x().getCUDAResPtr(), Q.y().getCUDAResPtr(),
      Q.z().getCUDAResPtr(), Q.w().getCUDAResPtr(), Q.x().getCUDAGradPtr(),
      Q.y().getCUDAGradPtr(), Q.z().getCUDAGradPtr(), Q.w().getCUDAGradPtr(),
      R(0, 0).getCUDAResPtr(), R(0, 1).getCUDAResPtr(), R(0, 2).getCUDAResPtr(),
      R(1, 0).getCUDAResPtr(), R(1, 1).getCUDAResPtr(), R(1, 2).getCUDAResPtr(),
      R(2, 0).getCUDAResPtr(), R(2, 1).getCUDAResPtr(), R(2, 2).getCUDAResPtr(),
      R(0, 0).getCUDAGradPtr(), R(0, 1).getCUDAGradPtr(),
      R(0, 2).getCUDAGradPtr(), R(1, 0).getCUDAGradPtr(),
      R(1, 1).getCUDAGradPtr(), R(1, 2).getCUDAGradPtr(),
      R(2, 0).getCUDAGradPtr(), R(2, 1).getCUDAGradPtr(),
      R(2, 2).getCUDAGradPtr());

  // TODO: use stream sync later
  cudaDeviceSynchronize();
  return R;
}

template <typename T>
JV4<T> RotationMatrixToQuaternion(const JM33<T> &R) {
  using W = R2Q_Address_Wrapper<T>;
  JV4<T> Q{};
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, CU_STREAM_NON_BLOCKING);
  const T *const address_R[3][3]{
      {R(0, 0).getCUDAResPtr(), R(0, 1).getCUDAResPtr(),
       R(0, 2).getCUDAResPtr()},
      {R(1, 0).getCUDAResPtr(), R(1, 1).getCUDAResPtr(),
       R(1, 2).getCUDAResPtr()},
      {R(2, 0).getCUDAResPtr(), R(2, 1).getCUDAResPtr(),
       R(2, 2).getCUDAResPtr()}};
  const T *const address_dR[3][3]{
      {R(0, 0).getCUDAGradPtr(), R(0, 1).getCUDAGradPtr(),
       R(0, 2).getCUDAGradPtr()},
      {R(1, 0).getCUDAGradPtr(), R(1, 1).getCUDAGradPtr(),
       R(1, 2).getCUDAGradPtr()},
      {R(2, 0).getCUDAGradPtr(), R(2, 1).getCUDAGradPtr(),
       R(2, 2).getCUDAGradPtr()}};
  cudaMemcpyToSymbolAsync(W::get_R(), address_R, 9 * sizeof(T *), 0,
                          cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(W::get_dR(), address_dR, 9 * sizeof(T *), 0,
                          cudaMemcpyHostToDevice, stream);
  const MegAutoBA::JetVector<T> &JV_Template = R(0, 0);
  for (int i = 0; i < 4; ++i) {
    Q(i).initAs(JV_Template);
  }

  T *const address_Q[4]{Q.x().getCUDAResPtr(), Q.y().getCUDAResPtr(),
                        Q.z().getCUDAResPtr(), Q.w().getCUDAResPtr()};
  T *const address_dQ[4]{Q.x().getCUDAGradPtr(), Q.y().getCUDAGradPtr(),
                         Q.z().getCUDAGradPtr(), Q.w().getCUDAGradPtr()};

  cudaMemcpyToSymbolAsync(W::get_Q(), address_Q, 4 * sizeof(T *), 0,
                          cudaMemcpyHostToDevice, stream);
  cudaMemcpyToSymbolAsync(W::get_dQ(), address_dQ, 4 * sizeof(T *), 0,
                          cudaMemcpyHostToDevice, stream);

  const auto nItem = JV_Template.getItemNum();
  const auto N = JV_Template.getGradShape();
  // 512 instead of 1024 for the limitation of registers
  dim3 block_dim(std::min(decltype(nItem)(512), nItem));
  dim3 grid_dim((nItem - 1) / block_dim.x + 1);

  RotationToQuaternion<T><<<grid_dim, block_dim, 0, stream>>>(nItem, N);

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return Q;
}

template <typename T>
JV4<T> &Normalize_(JV4<T> &Q) {
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, CU_STREAM_NON_BLOCKING);

  const auto nItem = Q(0).getItemNum();
  const auto N = Q(0).getGradShape();
  // 512 instead of 1024 for the limitation of registers
  dim3 block_dim(std::min(decltype(nItem)(768), nItem));
  dim3 grid_dim((nItem - 1) / block_dim.x + 1);
  Normalize_<T><<<grid_dim, block_dim, 0, stream>>>(
      nItem, N, Q.x().getCUDAResPtr(), Q.y().getCUDAResPtr(),
      Q.z().getCUDAResPtr(), Q.w().getCUDAResPtr(), Q.x().getCUDAGradPtr(),
      Q.y().getCUDAGradPtr(), Q.z().getCUDAGradPtr(), Q.w().getCUDAGradPtr());

  cudaStreamSynchronize(stream);
  cudaStreamDestroy(stream);
  return Q;
}

template JM33<float> QuaternionToRotationMatrix(const JV4<float> &Q);
template JM33<double> QuaternionToRotationMatrix(const JV4<double> &Q);
template JV4<float> RotationMatrixToQuaternion(const JM33<float> &R);
template JV4<double> RotationMatrixToQuaternion(const JM33<double> &R);
template JV4<float> &Normalize_(JV4<float> &Q);
template JV4<double> &Normalize_(JV4<double> &Q);
}  // namespace geo
}  // namespace MegAutoBA