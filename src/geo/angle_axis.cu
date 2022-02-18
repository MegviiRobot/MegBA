/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "geo/geo.cuh"
#include "wrapper.hpp"
#include "macro.h"

namespace MegBA {
namespace geo {
namespace {
template <typename T>
__global__ void AngleAxisToRotationKernel(
    const int nItem, const int N, const T *valueDevicePtr0,
    const T *valueDevicePtr1, const T *valueDevicePtr2, const T *gradDevicePtr0,
    const T *gradDevicePtr1, const T *gradDevicePtr2, T *R0, T *R1, T *R2,
    T *R3, T *R4, T *R5, T *R6, T *R7, T *R8, T *gradDevicePtrR0,
    T *gradDevicePtrR1, T *gradDevicePtrR2, T *gradDevicePtrR3,
    T *gradDevicePtrR4, T *gradDevicePtrR5, T *gradDevicePtrR6,
    T *gradDevicePtrR7, T *gradDevicePtrR8) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;
  const T angle_axis_x = valueDevicePtr0[idx];
  const T angle_axis_y = valueDevicePtr1[idx];
  const T angle_axis_z = valueDevicePtr2[idx];

  const T theta2 = angle_axis_x * angle_axis_x + angle_axis_y * angle_axis_y +
                   angle_axis_z * angle_axis_z;
  if (theta2 > std::numeric_limits<T>::epsilon()) {
    const T theta = Wrapper::sqrtG<T>::call(theta2);  // sqrt double
    const T wx = angle_axis_x / theta;
    const T wy = angle_axis_y / theta;
    const T wz = angle_axis_z / theta;

    T sintheta, costheta;

    Wrapper::sincosG<T>::call(theta, &sintheta, &costheta);
    const T one_minor_costheta = T(1.0) - costheta;
    const T wx_mul_one_minor_costheta = wx * one_minor_costheta;
    const T wy_mul_one_minor_costheta = wy * one_minor_costheta;
    const T wz_mul_one_minor_costheta = wz * one_minor_costheta;
    const T wx_mul_wy_mul_one_minor_costheta = wy * wx_mul_one_minor_costheta;
    const T wx_mul_wz_mul_one_minor_costheta = wz * wx_mul_one_minor_costheta;
    const T wy_mul_wz_mul_one_minor_costheta = wz * wy_mul_one_minor_costheta;
    const T wx_mul_sintheta = wx * sintheta;
    const T wy_mul_sintheta = wy * sintheta;
    const T wz_mul_sintheta = wz * sintheta;

    // clang-format on
    const T reciprocal_theta = 1 / theta;
    const T tmp1 = sintheta * reciprocal_theta;
    const T tmpwx = tmp1 * (wx * wx - T(1.0));
    const T tmpwy = tmp1 * (wy * wy - T(1.0));
    const T tmpwz = tmp1 * (wz * wz - T(1.0));

    for (int i = 0; i < N; ++i) {
      unsigned int index = idx + i * nItem;
      const T dv_angle_axis_x = gradDevicePtr0[index];
      const T dv_angle_axis_y = gradDevicePtr1[index];
      const T dv_angle_axis_z = gradDevicePtr2[index];

      const T dv_tmp1 =
          (angle_axis_x * dv_angle_axis_x + angle_axis_y * dv_angle_axis_y +
           angle_axis_z * dv_angle_axis_z);
      const T dv_theta = reciprocal_theta * dv_tmp1;

      const T dv_wx =
          reciprocal_theta *
          (dv_angle_axis_x - angle_axis_x * reciprocal_theta * dv_theta);
      const T dv_wy =
          reciprocal_theta *
          (dv_angle_axis_y - angle_axis_y * reciprocal_theta * dv_theta);
      const T dv_wz =
          reciprocal_theta *
          (dv_angle_axis_z - angle_axis_z * reciprocal_theta * dv_theta);

      gradDevicePtrR0[index] =
          tmpwx * dv_tmp1 + 2 * wx_mul_one_minor_costheta * dv_wx;
      gradDevicePtrR4[index] =
          tmpwy * dv_tmp1 + 2 * wy_mul_one_minor_costheta * dv_wy;
      gradDevicePtrR8[index] =
          tmpwz * dv_tmp1 + 2 * wz_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR1[index] =
          (wz * costheta + wx * wy_mul_sintheta) * dv_theta + sintheta * dv_wz +
          wy_mul_one_minor_costheta * dv_wx + wx_mul_one_minor_costheta * dv_wy;

      gradDevicePtrR5[index] =
          (wx * costheta + wy * wz_mul_sintheta) * dv_theta + sintheta * dv_wx +
          wz_mul_one_minor_costheta * dv_wy + wy_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR6[index] =
          (wy * costheta + wx * wz_mul_sintheta) * dv_theta + sintheta * dv_wy +
          wz_mul_one_minor_costheta * dv_wx + wx_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR2[index] =
          (-wy * costheta + wx * wz_mul_sintheta) * dv_theta -
          sintheta * dv_wy + wz_mul_one_minor_costheta * dv_wx +
          wx_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR3[index] =
          (-wz * costheta + wx * wy_mul_sintheta) * dv_theta -
          sintheta * dv_wz + wy_mul_one_minor_costheta * dv_wx +
          wx_mul_one_minor_costheta * dv_wy;

      gradDevicePtrR7[index] =
          (-wx * costheta + wy * wz_mul_sintheta) * dv_theta -
          sintheta * dv_wx + wz_mul_one_minor_costheta * dv_wy +
          wy_mul_one_minor_costheta * dv_wz;
    }

    R0[idx] = costheta + wx * wx_mul_one_minor_costheta;
    R1[idx] = wz_mul_sintheta + wx_mul_wy_mul_one_minor_costheta;
    R2[idx] = -wy_mul_sintheta + wx_mul_wz_mul_one_minor_costheta;

    R3[idx] = -wz_mul_sintheta + wx_mul_wy_mul_one_minor_costheta;
    R4[idx] = costheta + wy * wy_mul_one_minor_costheta;
    R5[idx] = wx_mul_sintheta + wy_mul_wz_mul_one_minor_costheta;

    R6[idx] = wy_mul_sintheta + wx_mul_wz_mul_one_minor_costheta;
    R7[idx] = -wx_mul_sintheta + wy_mul_wz_mul_one_minor_costheta;
    R8[idx] = costheta + wz * wz_mul_one_minor_costheta;
  } else {
    // Near zero, we switch to using the first order Taylor expansion.
    for (int i = 0; i < N; ++i) {
      unsigned int index = idx + i * nItem;
      const T dv_angle_axis_x = gradDevicePtr0[index];
      const T dv_angle_axis_y = gradDevicePtr1[index];
      const T dv_angle_axis_z = gradDevicePtr2[index];
      gradDevicePtrR0[index] = 0;
      gradDevicePtrR1[index] = dv_angle_axis_z;
      gradDevicePtrR2[index] = -dv_angle_axis_y;
      gradDevicePtrR3[index] = -dv_angle_axis_z;
      gradDevicePtrR4[index] = 0;
      gradDevicePtrR5[index] = dv_angle_axis_x;
      gradDevicePtrR6[index] = dv_angle_axis_y;
      gradDevicePtrR7[index] = -dv_angle_axis_x;
      gradDevicePtrR8[index] = 0;
    }
    R0[idx] = T(1.0);
    R1[idx] = angle_axis_z;
    R2[idx] = -angle_axis_y;

    R3[idx] = -angle_axis_z;
    R4[idx] = T(1.0);
    R5[idx] = angle_axis_x;

    R6[idx] = angle_axis_y;
    R7[idx] = -angle_axis_x;
    R8[idx] = T(1.0);
  }
}

template <typename T>
__global__ void AngleAxisToRotationKernelFastGradKernel(
    const int nItem, const int N, const T *valueDevicePtr0,
    const T *valueDevicePtr1, const T *valueDevicePtr2,
    const int grad_position0, const int grad_position1,
    const int grad_position2, T *R0, T *R1, T *R2, T *R3, T *R4, T *R5, T *R6,
    T *R7, T *R8, T *gradDevicePtrR0, T *gradDevicePtrR1, T *gradDevicePtrR2,
    T *gradDevicePtrR3, T *gradDevicePtrR4, T *gradDevicePtrR5,
    T *gradDevicePtrR6, T *gradDevicePtrR7, T *gradDevicePtrR8) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;
  const T angle_axis_x = valueDevicePtr0[idx];
  const T angle_axis_y = valueDevicePtr1[idx];
  const T angle_axis_z = valueDevicePtr2[idx];

  const T theta2 = angle_axis_x * angle_axis_x + angle_axis_y * angle_axis_y +
                   angle_axis_z * angle_axis_z;
  if (theta2 > std::numeric_limits<T>::epsilon()) {
    const T theta = Wrapper::sqrtG<T>::call(theta2);  // sqrt double
    const T wx = angle_axis_x / theta;
    const T wy = angle_axis_y / theta;
    const T wz = angle_axis_z / theta;

    T sintheta, costheta;

    Wrapper::sincosG<T>::call(theta, &sintheta, &costheta);
    const T one_minor_costheta = T(1.0) - costheta;
    const T wx_mul_one_minor_costheta = wx * one_minor_costheta;
    const T wy_mul_one_minor_costheta = wy * one_minor_costheta;
    const T wz_mul_one_minor_costheta = wz * one_minor_costheta;
    const T wx_mul_wy_mul_one_minor_costheta = wy * wx_mul_one_minor_costheta;
    const T wx_mul_wz_mul_one_minor_costheta = wz * wx_mul_one_minor_costheta;
    const T wy_mul_wz_mul_one_minor_costheta = wz * wy_mul_one_minor_costheta;
    const T wx_mul_sintheta = wx * sintheta;
    const T wy_mul_sintheta = wy * sintheta;
    const T wz_mul_sintheta = wz * sintheta;

    const T reciprocal_theta = 1 / theta;
    const T tmp1 = sintheta * reciprocal_theta;
    const T tmpwx = tmp1 * (wx * wx - T(1.0));
    const T tmpwy = tmp1 * (wy * wy - T(1.0));
    const T tmpwz = tmp1 * (wz * wz - T(1.0));

    for (int i = 0; i < N; ++i) {
      unsigned int index = idx + i * nItem;
      const T dv_angle_axis_x = i == grad_position0 ? 1 : 0;
      const T dv_angle_axis_y = i == grad_position1 ? 1 : 0;
      const T dv_angle_axis_z = i == grad_position2 ? 1 : 0;

      const T dv_tmp1 =
          (angle_axis_x * dv_angle_axis_x + angle_axis_y * dv_angle_axis_y +
           angle_axis_z * dv_angle_axis_z);
      const T dv_theta = reciprocal_theta * dv_tmp1;

      const T dv_wx =
          reciprocal_theta *
          (dv_angle_axis_x - angle_axis_x * reciprocal_theta * dv_theta);
      const T dv_wy =
          reciprocal_theta *
          (dv_angle_axis_y - angle_axis_y * reciprocal_theta * dv_theta);
      const T dv_wz =
          reciprocal_theta *
          (dv_angle_axis_z - angle_axis_z * reciprocal_theta * dv_theta);

      gradDevicePtrR0[index] =
          tmpwx * dv_tmp1 + 2 * wx_mul_one_minor_costheta * dv_wx;
      gradDevicePtrR4[index] =
          tmpwy * dv_tmp1 + 2 * wy_mul_one_minor_costheta * dv_wy;
      gradDevicePtrR8[index] =
          tmpwz * dv_tmp1 + 2 * wz_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR1[index] =
          (wz * costheta + wx * wy_mul_sintheta) * dv_theta + sintheta * dv_wz +
          wy_mul_one_minor_costheta * dv_wx + wx_mul_one_minor_costheta * dv_wy;

      gradDevicePtrR5[index] =
          (wx * costheta + wy * wz_mul_sintheta) * dv_theta + sintheta * dv_wx +
          wz_mul_one_minor_costheta * dv_wy + wy_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR6[index] =
          (wy * costheta + wx * wz_mul_sintheta) * dv_theta + sintheta * dv_wy +
          wz_mul_one_minor_costheta * dv_wx + wx_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR2[index] =
          (-wy * costheta + wx * wz_mul_sintheta) * dv_theta -
          sintheta * dv_wy + wz_mul_one_minor_costheta * dv_wx +
          wx_mul_one_minor_costheta * dv_wz;

      gradDevicePtrR3[index] =
          (-wz * costheta + wx * wy_mul_sintheta) * dv_theta -
          sintheta * dv_wz + wy_mul_one_minor_costheta * dv_wx +
          wx_mul_one_minor_costheta * dv_wy;

      gradDevicePtrR7[index] =
          (-wx * costheta + wy * wz_mul_sintheta) * dv_theta -
          sintheta * dv_wx + wz_mul_one_minor_costheta * dv_wy +
          wy_mul_one_minor_costheta * dv_wz;
    }

    R0[idx] = costheta + wx * wx_mul_one_minor_costheta;
    R1[idx] = wz_mul_sintheta + wx_mul_wy_mul_one_minor_costheta;
    R2[idx] = -wy_mul_sintheta + wx_mul_wz_mul_one_minor_costheta;

    R3[idx] = -wz_mul_sintheta + wx_mul_wy_mul_one_minor_costheta;
    R4[idx] = costheta + wy * wy_mul_one_minor_costheta;
    R5[idx] = wx_mul_sintheta + wy_mul_wz_mul_one_minor_costheta;

    R6[idx] = wy_mul_sintheta + wx_mul_wz_mul_one_minor_costheta;
    R7[idx] = -wx_mul_sintheta + wy_mul_wz_mul_one_minor_costheta;
    R8[idx] = costheta + wz * wz_mul_one_minor_costheta;
  } else {
    // Near zero, we switch to using the first order Taylor expansion.
    for (int i = 0; i < N; ++i) {
      unsigned int index = idx + i * nItem;
      const T dv_angle_axis_x = i == grad_position0 ? 1 : 0;
      const T dv_angle_axis_y = i == grad_position1 ? 1 : 0;
      const T dv_angle_axis_z = i == grad_position2 ? 1 : 0;
      gradDevicePtrR0[index] = 0;
      gradDevicePtrR1[index] = dv_angle_axis_z;
      gradDevicePtrR2[index] = -dv_angle_axis_y;
      gradDevicePtrR3[index] = -dv_angle_axis_z;
      gradDevicePtrR4[index] = 0;
      gradDevicePtrR5[index] = dv_angle_axis_x;
      gradDevicePtrR6[index] = dv_angle_axis_y;
      gradDevicePtrR7[index] = -dv_angle_axis_x;
      gradDevicePtrR8[index] = 0;
    }
    R0[idx] = T(1.0);
    R1[idx] = angle_axis_z;
    R2[idx] = -angle_axis_y;

    R3[idx] = -angle_axis_z;
    R4[idx] = T(1.0);
    R5[idx] = angle_axis_x;

    R6[idx] = angle_axis_y;
    R7[idx] = -angle_axis_x;
    R8[idx] = T(1.0);
  }
}
}

        template <typename T>
JM33<T> AngleAxisToRotationKernelMatrix(const JV3<T> &AxisAngle) {
  JM33<T> R{};
  const MegBA::JetVector<T> &JV_Template = AxisAngle(0, 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R(i, j).initAs(JV_Template);
    }
  }

  bool use_fast_grad{true};
  for (int i = 0; i < 3; ++i)
    use_fast_grad &= AxisAngle(i).getGradPosition() != -1;

  const auto N = JV_Template.getGradShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = JV_Template.getItemNum(i);
    // 512 instead of 1024 for the limitation of registers
    dim3 block_dim(std::min(decltype(nItem)(512), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    ASSERT_CUDA_NO_ERROR();

    if (use_fast_grad)
      AngleAxisToRotationKernelFastGradKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, AxisAngle(0).getCUDAResPtr()[i],
          AxisAngle(1).getCUDAResPtr()[i], AxisAngle(2).getCUDAResPtr()[i],
          AxisAngle(0).getGradPosition(), AxisAngle(1).getGradPosition(),
          AxisAngle(2).getGradPosition(), R(0, 0).getCUDAResPtr()[i],
          R(1, 0).getCUDAResPtr()[i], R(2, 0).getCUDAResPtr()[i],
          R(0, 1).getCUDAResPtr()[i], R(1, 1).getCUDAResPtr()[i],
          R(2, 1).getCUDAResPtr()[i], R(0, 2).getCUDAResPtr()[i],
          R(1, 2).getCUDAResPtr()[i], R(2, 2).getCUDAResPtr()[i],
          R(0, 0).getCUDAGradPtr()[i], R(1, 0).getCUDAGradPtr()[i],
          R(2, 0).getCUDAGradPtr()[i], R(0, 1).getCUDAGradPtr()[i],
          R(1, 1).getCUDAGradPtr()[i], R(2, 1).getCUDAGradPtr()[i],
          R(0, 2).getCUDAGradPtr()[i], R(1, 2).getCUDAGradPtr()[i],
          R(2, 2).getCUDAGradPtr()[i]);
    else
      AngleAxisToRotationKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, AxisAngle(0, 0).getCUDAResPtr()[i],
          AxisAngle(1, 0).getCUDAResPtr()[i],
          AxisAngle(2, 0).getCUDAResPtr()[i],
          AxisAngle(0, 0).getCUDAGradPtr()[i],
          AxisAngle(1, 0).getCUDAGradPtr()[i],
          AxisAngle(2, 0).getCUDAGradPtr()[i], R(0, 0).getCUDAResPtr()[i],
          R(1, 0).getCUDAResPtr()[i], R(2, 0).getCUDAResPtr()[i],
          R(0, 1).getCUDAResPtr()[i], R(1, 1).getCUDAResPtr()[i],
          R(2, 1).getCUDAResPtr()[i], R(0, 2).getCUDAResPtr()[i],
          R(1, 2).getCUDAResPtr()[i], R(2, 2).getCUDAResPtr()[i],
          R(0, 0).getCUDAGradPtr()[i], R(1, 0).getCUDAGradPtr()[i],
          R(2, 0).getCUDAGradPtr()[i], R(0, 1).getCUDAGradPtr()[i],
          R(1, 1).getCUDAGradPtr()[i], R(2, 1).getCUDAGradPtr()[i],
          R(0, 2).getCUDAGradPtr()[i], R(1, 2).getCUDAGradPtr()[i],
          R(2, 2).getCUDAGradPtr()[i]);
    ASSERT_CUDA_NO_ERROR();
  }

  return R;
}

template <typename T>
JM33<T> AngleAxisToRotationKernelMatrix(
    const Eigen::Map<const JVD<T>> &AxisAngle) {
  JM33<T> R{};
  const MegBA::JetVector<T> &JV_Template = AxisAngle(0, 0);
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 3; ++j) {
      R(i, j).initAs(JV_Template);
    }
  }

  bool use_fast_grad{true};
  for (int i = 0; i < 3; ++i)
    use_fast_grad &= AxisAngle(i).getGradPosition() != -1;

  const auto N = JV_Template.getGradShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = JV_Template.getItemNum(i);
    // 512 instead of 1024 for the limitation of registers
    dim3 block_dim(std::min(decltype(nItem)(512), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);
    ASSERT_CUDA_NO_ERROR();

    if (use_fast_grad)
      AngleAxisToRotationKernelFastGradKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, AxisAngle(0).getCUDAResPtr()[i],
          AxisAngle(1).getCUDAResPtr()[i], AxisAngle(2).getCUDAResPtr()[i],
          AxisAngle(0).getGradPosition(), AxisAngle(1).getGradPosition(),
          AxisAngle(2).getGradPosition(), R(0, 0).getCUDAResPtr()[i],
          R(1, 0).getCUDAResPtr()[i], R(2, 0).getCUDAResPtr()[i],
          R(0, 1).getCUDAResPtr()[i], R(1, 1).getCUDAResPtr()[i],
          R(2, 1).getCUDAResPtr()[i], R(0, 2).getCUDAResPtr()[i],
          R(1, 2).getCUDAResPtr()[i], R(2, 2).getCUDAResPtr()[i],
          R(0, 0).getCUDAGradPtr()[i], R(1, 0).getCUDAGradPtr()[i],
          R(2, 0).getCUDAGradPtr()[i], R(0, 1).getCUDAGradPtr()[i],
          R(1, 1).getCUDAGradPtr()[i], R(2, 1).getCUDAGradPtr()[i],
          R(0, 2).getCUDAGradPtr()[i], R(1, 2).getCUDAGradPtr()[i],
          R(2, 2).getCUDAGradPtr()[i]);
    else
      AngleAxisToRotationKernel<T><<<grid_dim, block_dim>>>(
          nItem, N, AxisAngle(0, 0).getCUDAResPtr()[i],
          AxisAngle(1, 0).getCUDAResPtr()[i],
          AxisAngle(2, 0).getCUDAResPtr()[i],
          AxisAngle(0, 0).getCUDAGradPtr()[i],
          AxisAngle(1, 0).getCUDAGradPtr()[i],
          AxisAngle(2, 0).getCUDAGradPtr()[i], R(0, 0).getCUDAResPtr()[i],
          R(1, 0).getCUDAResPtr()[i], R(2, 0).getCUDAResPtr()[i],
          R(0, 1).getCUDAResPtr()[i], R(1, 1).getCUDAResPtr()[i],
          R(2, 1).getCUDAResPtr()[i], R(0, 2).getCUDAResPtr()[i],
          R(1, 2).getCUDAResPtr()[i], R(2, 2).getCUDAResPtr()[i],
          R(0, 0).getCUDAGradPtr()[i], R(1, 0).getCUDAGradPtr()[i],
          R(2, 0).getCUDAGradPtr()[i], R(0, 1).getCUDAGradPtr()[i],
          R(1, 1).getCUDAGradPtr()[i], R(2, 1).getCUDAGradPtr()[i],
          R(0, 2).getCUDAGradPtr()[i], R(1, 2).getCUDAGradPtr()[i],
          R(2, 2).getCUDAGradPtr()[i]);
    ASSERT_CUDA_NO_ERROR();
  }

  return R;
}

template JM33<float> AngleAxisToRotationKernelMatrix(
    const JV3<float> &AxisAngle);
template JM33<double> AngleAxisToRotationKernelMatrix(
    const JV3<double> &AxisAngle);

template JM33<float> AngleAxisToRotationKernelMatrix(
    const Eigen::Map<const JVD<float>> &AxisAngle);
template JM33<double> AngleAxisToRotationKernelMatrix(
    const Eigen::Map<const JVD<double>> &AxisAngle);
}
}