/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "geo/geo.cuh"
#include "macro.h"
#include "wrapper.hpp"

namespace MegBA {
namespace geo {
namespace {
template <typename T>
__device__ void AngleAxisRotatePoint(
    const T angle_axis0, const T angle_axis1, const T angle_axis2,
    const T point0, const T point1, const T point2, T &re_projection0,
    T &re_projection1, T &re_projection2, T *dprojection0_dangleaxis,
    T *dprojection1_dangleaxis, T *dprojection2_dangleaxis,
    T *dprojection0_dpoint, T *dprojection1_dpoint, T *dprojection2_dpoint) {
  const T theta2 = angle_axis0 * angle_axis0 + angle_axis1 * angle_axis1 +
                   angle_axis2 * angle_axis2;
  if (theta2 > std::numeric_limits<T>::epsilon()) {
    const T theta = Wrapper::sqrtG<T>::call(theta2);
    T sintheta, costheta;
    Wrapper::sincosG<T>::call(theta, &sintheta, &costheta);
    const T theta_inverse = T(1.0) / theta;
    const T negative_theta_inverse_pow2 = -theta_inverse * theta_inverse;
    const T one_minus_costheta = T(1.0) - costheta;
    const T w[3] = {angle_axis0 * theta_inverse, angle_axis1 * theta_inverse,
                    angle_axis2 * theta_inverse};
    const T w_cross_pt[3] = {w[1] * point2 - w[2] * point1,
                             w[2] * point0 - w[0] * point2,
                             w[0] * point1 - w[1] * point0};
    const T tmp =
        (w[0] * point0 + w[1] * point1 + w[2] * point2) * one_minus_costheta;
    re_projection0 = point0 * costheta + w_cross_pt[0] * sintheta + w[0] * tmp;
    re_projection1 = point1 * costheta + w_cross_pt[1] * sintheta + w[1] * tmp;
    re_projection2 = point2 * costheta + w_cross_pt[2] * sintheta + w[2] * tmp;

    const T dtheta_daa[3] = {theta_inverse * angle_axis0,
                             theta_inverse * angle_axis1,
                             theta_inverse * angle_axis2};
    const T dw_daa[9] = {
        theta_inverse +
            negative_theta_inverse_pow2 * angle_axis0 * dtheta_daa[0],
        negative_theta_inverse_pow2 * angle_axis0 * dtheta_daa[1],
        negative_theta_inverse_pow2 * angle_axis0 * dtheta_daa[2],
        negative_theta_inverse_pow2 * angle_axis1 * dtheta_daa[0],
        theta_inverse +
            negative_theta_inverse_pow2 * angle_axis1 * dtheta_daa[1],
        negative_theta_inverse_pow2 * angle_axis1 * dtheta_daa[2],
        negative_theta_inverse_pow2 * angle_axis2 * dtheta_daa[0],
        negative_theta_inverse_pow2 * angle_axis2 * dtheta_daa[1],
        theta_inverse +
            negative_theta_inverse_pow2 * angle_axis2 * dtheta_daa[2]};
    const T dwcrosspt_dx[9] = {0, -w[2], w[1], w[2], 0, -w[0], -w[1], w[0], 0};
    const T dwcrosspt_daa[9] = {point2 * dw_daa[3] - point1 * dw_daa[6],
                                point2 * dw_daa[4] - point1 * dw_daa[7],
                                point2 * dw_daa[5] - point1 * dw_daa[8],
                                -point2 * dw_daa[0] + point0 * dw_daa[6],
                                -point2 * dw_daa[1] + point0 * dw_daa[7],
                                -point2 * dw_daa[2] + point0 * dw_daa[8],
                                point1 * dw_daa[0] - point0 * dw_daa[3],
                                point1 * dw_daa[1] - point0 * dw_daa[4],
                                point1 * dw_daa[2] - point0 * dw_daa[5]};
    const T dtmp_dx[3] = {one_minus_costheta * w[0], one_minus_costheta * w[1],
                          one_minus_costheta * w[2]};
    const T dtmp_daa[3] = {
        sintheta * (w[0] * point0 + w[1] * point1 + w[2] * point2) *
                dtheta_daa[0] +
            one_minus_costheta *
                (point0 * dw_daa[0] + point1 * dw_daa[3] + point2 * dw_daa[6]),
        sintheta * (w[0] * point0 + w[1] * point1 + w[2] * point2) *
                dtheta_daa[1] +
            one_minus_costheta *
                (point0 * dw_daa[1] + point1 * dw_daa[4] + point2 * dw_daa[7]),
        sintheta * (w[0] * point0 + w[1] * point1 + w[2] * point2) *
                dtheta_daa[2] +
            one_minus_costheta *
                (point0 * dw_daa[2] + point1 * dw_daa[5] + point2 * dw_daa[8])};
    const T dcostheta_daa[3] = {-sintheta * dtheta_daa[0],
                                -sintheta * dtheta_daa[1],
                                -sintheta * dtheta_daa[2]};
    const T dsintheta_daa[3] = {costheta * dtheta_daa[0],
                                costheta * dtheta_daa[1],
                                costheta * dtheta_daa[2]};

    dprojection0_dangleaxis[0] =
        point0 * dcostheta_daa[0] + w_cross_pt[0] * dsintheta_daa[0] +
        sintheta * dwcrosspt_daa[0] + w[0] * dtmp_daa[0] + tmp * dw_daa[0];
    dprojection0_dangleaxis[1] =
        point0 * dcostheta_daa[1] + w_cross_pt[0] * dsintheta_daa[1] +
        sintheta * dwcrosspt_daa[1] + w[0] * dtmp_daa[1] + tmp * dw_daa[1];
    dprojection0_dangleaxis[2] =
        point0 * dcostheta_daa[2] + w_cross_pt[0] * dsintheta_daa[2] +
        sintheta * dwcrosspt_daa[2] + w[0] * dtmp_daa[2] + tmp * dw_daa[2];
    dprojection1_dangleaxis[0] =
        point1 * dcostheta_daa[0] + w_cross_pt[1] * dsintheta_daa[0] +
        sintheta * dwcrosspt_daa[3] + w[1] * dtmp_daa[0] + tmp * dw_daa[3];
    dprojection1_dangleaxis[1] =
        point1 * dcostheta_daa[1] + w_cross_pt[1] * dsintheta_daa[1] +
        sintheta * dwcrosspt_daa[4] + w[1] * dtmp_daa[1] + tmp * dw_daa[4];
    dprojection1_dangleaxis[2] =
        point1 * dcostheta_daa[2] + w_cross_pt[1] * dsintheta_daa[2] +
        sintheta * dwcrosspt_daa[5] + w[1] * dtmp_daa[2] + tmp * dw_daa[5];
    dprojection2_dangleaxis[0] =
        point2 * dcostheta_daa[0] + w_cross_pt[2] * dsintheta_daa[0] +
        sintheta * dwcrosspt_daa[6] + w[2] * dtmp_daa[0] + tmp * dw_daa[6];
    dprojection2_dangleaxis[1] =
        point2 * dcostheta_daa[1] + w_cross_pt[2] * dsintheta_daa[1] +
        sintheta * dwcrosspt_daa[7] + w[2] * dtmp_daa[1] + tmp * dw_daa[7];
    dprojection2_dangleaxis[2] =
        point2 * dcostheta_daa[2] + w_cross_pt[2] * dsintheta_daa[2] +
        sintheta * dwcrosspt_daa[8] + w[2] * dtmp_daa[2] + tmp * dw_daa[8];

    dprojection0_dpoint[0] =
        costheta + sintheta * dwcrosspt_dx[0] + w[0] * dtmp_dx[0];
    dprojection0_dpoint[1] = sintheta * dwcrosspt_dx[1] + w[0] * dtmp_dx[1];
    dprojection0_dpoint[2] = sintheta * dwcrosspt_dx[2] + w[0] * dtmp_dx[2];
    dprojection1_dpoint[0] = sintheta * dwcrosspt_dx[3] + w[1] * dtmp_dx[0];
    dprojection1_dpoint[1] =
        costheta + sintheta * dwcrosspt_dx[4] + w[1] * dtmp_dx[1];
    dprojection1_dpoint[2] = sintheta * dwcrosspt_dx[5] + w[1] * dtmp_dx[2];
    dprojection2_dpoint[0] = sintheta * dwcrosspt_dx[6] + w[2] * dtmp_dx[0];
    dprojection2_dpoint[1] = sintheta * dwcrosspt_dx[7] + w[2] * dtmp_dx[1];
    dprojection2_dpoint[2] =
        costheta + sintheta * dwcrosspt_dx[8] + w[2] * dtmp_dx[2];

  } else {
    const T w_cross_pt[3] = {angle_axis1 * point2 - angle_axis2 * point1,
                             angle_axis2 * point0 - angle_axis0 * point2,
                             angle_axis0 * point1 - angle_axis1 * point0};
    re_projection0 = point0 + w_cross_pt[0];
    re_projection1 = point1 + w_cross_pt[1];
    re_projection2 = point2 + w_cross_pt[2];

    dprojection0_dangleaxis[0] = 0;
    dprojection0_dangleaxis[1] = point2;
    dprojection0_dangleaxis[2] = -point1;
    dprojection1_dangleaxis[0] = -point2;
    dprojection1_dangleaxis[1] = 0;
    dprojection1_dangleaxis[2] = point0;
    dprojection2_dangleaxis[0] = point1;
    dprojection2_dangleaxis[1] = -point0;
    dprojection2_dangleaxis[2] = 0;

    dprojection0_dpoint[0] = 1;
    dprojection0_dpoint[1] = -angle_axis2;
    dprojection0_dpoint[2] = angle_axis1;
    dprojection1_dpoint[0] = angle_axis2;
    dprojection1_dpoint[1] = 1;
    dprojection1_dpoint[2] = -angle_axis0;
    dprojection2_dpoint[0] = -angle_axis1;
    dprojection2_dpoint[1] = angle_axis0;
    dprojection2_dpoint[2] = 1;
  }
}

template <typename T>
__global__ void AnalyticalDerivativesKernelGradKernel(
    const int nItem, const int N, const T *angle_axis0_ptr,
    const T *angle_axis1_ptr, const T *angle_axis2_ptr, const T *t0_ptr,
    const T *t1_ptr, const T *t2_ptr, const T *f_ptr, const T *k1_ptr,
    const T *k2_ptr, const T *point0_ptr, const T *point1_ptr,
    const T *point2_ptr, const T *obs0_ptr, const T *obs1_ptr,
    T *error0_valueDevicePtr, T *error1_valueDevicePtr, T *gradDevicePtrError0,
    T *gradDevicePtrError1) {
  unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
  if (idx >= nItem) return;
  T dprojection0_dangleaxis[3];
  T dprojection1_dangleaxis[3];
  T dprojection2_dangleaxis[3];
  T dprojection0_dpoint[3];
  T dprojection1_dpoint[3];
  T dprojection2_dpoint[3];
  T re_projection0, re_projection1, re_projection2;
  AngleAxisRotatePoint(
      angle_axis0_ptr[idx], angle_axis1_ptr[idx], angle_axis2_ptr[idx],
      point0_ptr[idx], point1_ptr[idx], point2_ptr[idx], re_projection0,
      re_projection1, re_projection2, dprojection0_dangleaxis,
      dprojection1_dangleaxis, dprojection2_dangleaxis, dprojection0_dpoint,
      dprojection1_dpoint, dprojection2_dpoint);

  re_projection0 += t0_ptr[idx];
  re_projection1 += t1_ptr[idx];
  re_projection2 += t2_ptr[idx];
  const T B0 = -re_projection0 / re_projection2;
  const T B1 = -re_projection1 / re_projection2;
  const T l2_pow2 = B0 * B0 + B1 * B1;
  const T rp = T(1.) + k1_ptr[idx] * l2_pow2 + k2_ptr[idx] * l2_pow2 * l2_pow2;
  const T fr = f_ptr[idx] * rp;
  error0_valueDevicePtr[idx] = fr * B0 - obs0_ptr[idx];
  error1_valueDevicePtr[idx] = fr * B1 - obs1_ptr[idx];

  const T tmp0 = f_ptr[idx] * B0 * (k1_ptr[idx] + 2 * k2_ptr[idx] * l2_pow2);
  const T tmp1 = f_ptr[idx] * B1 * (k1_ptr[idx] + 2 * k2_ptr[idx] * l2_pow2);
  const T negative_reciprocal_p2 = -1 / re_projection2;
  const T p0_dividedby_p2pow2 =
      re_projection0 / (re_projection2 * re_projection2);
  const T p1_dividedby_p2pow2 =
      re_projection1 / (re_projection2 * re_projection2);

  T error_factor0 = negative_reciprocal_p2 * (fr + 2 * tmp0 * B0);
  T error_factor1 = negative_reciprocal_p2 * 2 * tmp0 * B1;
  T error_factor2 = p0_dividedby_p2pow2 * (fr + 2 * tmp0 * B0) +
                    p1_dividedby_p2pow2 * 2 * tmp0 * B1;
  // derror0 / dangle_axis
  gradDevicePtrError0[idx] = error_factor0 * dprojection0_dangleaxis[0] +
                             error_factor1 * dprojection1_dangleaxis[0] +
                             error_factor2 * dprojection2_dangleaxis[0];
  gradDevicePtrError0[idx + 1 * nItem] =
      error_factor0 * dprojection0_dangleaxis[1] +
      error_factor1 * dprojection1_dangleaxis[1] +
      error_factor2 * dprojection2_dangleaxis[1];
  gradDevicePtrError0[idx + 2 * nItem] =
      error_factor0 * dprojection0_dangleaxis[2] +
      error_factor1 * dprojection1_dangleaxis[2] +
      error_factor2 * dprojection2_dangleaxis[2];
  // derror0 / dt
  gradDevicePtrError0[idx + 3 * nItem] = error_factor0;
  gradDevicePtrError0[idx + 4 * nItem] = error_factor1;
  gradDevicePtrError0[idx + 5 * nItem] = error_factor2;
  // derror0 / df
  gradDevicePtrError0[idx + 6 * nItem] = rp * B0;
  // derror0 / dk1
  gradDevicePtrError0[idx + 7 * nItem] = f_ptr[idx] * l2_pow2 * B0;
  // derror0 / dk2
  gradDevicePtrError0[idx + 8 * nItem] = f_ptr[idx] * l2_pow2 * l2_pow2 * B0;
  // derror0 / dpoint_xyz
  gradDevicePtrError0[idx + 9 * nItem] =
      error_factor0 * dprojection0_dpoint[0] +
      error_factor1 * dprojection1_dpoint[0] +
      error_factor2 * dprojection2_dpoint[0];
  gradDevicePtrError0[idx + 10 * nItem] =
      error_factor0 * dprojection0_dpoint[1] +
      error_factor1 * dprojection1_dpoint[1] +
      error_factor2 * dprojection2_dpoint[1];
  gradDevicePtrError0[idx + 11 * nItem] =
      error_factor0 * dprojection0_dpoint[2] +
      error_factor1 * dprojection1_dpoint[2] +
      error_factor2 * dprojection2_dpoint[2];

  // -------------------------------------------------------------------------------------
  error_factor0 = negative_reciprocal_p2 * 2 * tmp1 * B0;
  error_factor1 = negative_reciprocal_p2 * (fr + 2 * tmp1 * B1);
  error_factor2 = p0_dividedby_p2pow2 * 2 * tmp1 * B0 +
                  p1_dividedby_p2pow2 * (fr + 2 * tmp1 * B1);
  // derror0 / dangle_axis
  gradDevicePtrError1[idx] = error_factor0 * dprojection0_dangleaxis[0] +
                             error_factor1 * dprojection1_dangleaxis[0] +
                             error_factor2 * dprojection2_dangleaxis[0];
  gradDevicePtrError1[idx + 1 * nItem] =
      error_factor0 * dprojection0_dangleaxis[1] +
      error_factor1 * dprojection1_dangleaxis[1] +
      error_factor2 * dprojection2_dangleaxis[1];
  gradDevicePtrError1[idx + 2 * nItem] =
      error_factor0 * dprojection0_dangleaxis[2] +
      error_factor1 * dprojection1_dangleaxis[2] +
      error_factor2 * dprojection2_dangleaxis[2];
  // derror1 / dt
  gradDevicePtrError1[idx + 3 * nItem] = error_factor0;
  gradDevicePtrError1[idx + 4 * nItem] = error_factor1;
  gradDevicePtrError1[idx + 5 * nItem] = error_factor2;
  // derror1 / df
  gradDevicePtrError1[idx + 6 * nItem] = rp * B1;
  // derror1 / dk1
  gradDevicePtrError1[idx + 7 * nItem] = f_ptr[idx] * l2_pow2 * B1;
  // derror1 / dk2
  gradDevicePtrError1[idx + 8 * nItem] = f_ptr[idx] * l2_pow2 * l2_pow2 * B1;
  // derror1 / dpoint_xyz
  gradDevicePtrError1[idx + 9 * nItem] =
      error_factor0 * dprojection0_dpoint[0] +
      error_factor1 * dprojection1_dpoint[0] +
      error_factor2 * dprojection2_dpoint[0];
  gradDevicePtrError1[idx + 10 * nItem] =
      error_factor0 * dprojection0_dpoint[1] +
      error_factor1 * dprojection1_dpoint[1] +
      error_factor2 * dprojection2_dpoint[1];
  gradDevicePtrError1[idx + 11 * nItem] =
      error_factor0 * dprojection0_dpoint[2] +
      error_factor1 * dprojection1_dpoint[2] +
      error_factor2 * dprojection2_dpoint[2];
}
}  // namespace

template <typename T>
MegBA::geo::JVD<T> AnalyticalDerivativesKernelMatrix(
    const Eigen::Map<const JVD<T>> &AxisAngle,
    const Eigen::Map<const JVD<T>> &t,
    const Eigen::Map<const JVD<T>> &intrinsics, const JVD<T> &point_xyz,
    const JVD<T> &obs_uv) {
  const MegBA::JetVector<T> &JV_Template = AxisAngle(0, 0);
  MegBA::geo::JVD<T> error{};
  error.resize(2, 1);
  for (int i = 0; i < 2; ++i) {
    error(i).initAs(JV_Template);
  }
  const auto N = JV_Template.getGradShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = JV_Template.getItemNum(i);
    // 512 instead of 1024 for the limitation of registers
    dim3 block_dim(std::min(decltype(nItem)(512), nItem));
    dim3 grid_dim((nItem - 1) / block_dim.x + 1);

    AnalyticalDerivativesKernelGradKernel<T><<<grid_dim, block_dim>>>(
        nItem, N, AxisAngle(0).getCUDAResPtr()[i],
        AxisAngle(1).getCUDAResPtr()[i], AxisAngle(2).getCUDAResPtr()[i],
        t(0).getCUDAResPtr()[i], t(1).getCUDAResPtr()[i],
        t(2).getCUDAResPtr()[i], intrinsics(0).getCUDAResPtr()[i],
        intrinsics(1).getCUDAResPtr()[i], intrinsics(2).getCUDAResPtr()[i],
        point_xyz(0).getCUDAResPtr()[i], point_xyz(1).getCUDAResPtr()[i],
        point_xyz(2).getCUDAResPtr()[i], obs_uv(0).getCUDAResPtr()[i],
        obs_uv(1).getCUDAResPtr()[i], error(0).getCUDAResPtr()[i],
        error(1).getCUDAResPtr()[i], error(0).getCUDAGradPtr()[i],
        error(1).getCUDAGradPtr()[i]);
    ASSERT_CUDA_NO_ERROR();
  }
  return error;
}

template MegBA::geo::JVD<float> AnalyticalDerivativesKernelMatrix(
    const Eigen::Map<const JVD<float>> &AxisAngle,
    const Eigen::Map<const JVD<float>> &t,
    const Eigen::Map<const JVD<float>> &intrinsics, const JVD<float> &point_xyz,
    const JVD<float> &obs_uv);

template MegBA::geo::JVD<double> AnalyticalDerivativesKernelMatrix(
    const Eigen::Map<const JVD<double>> &AxisAngle,
    const Eigen::Map<const JVD<double>> &t,
    const Eigen::Map<const JVD<double>> &intrinsics,
    const JVD<double> &point_xyz, const JVD<double> &obs_uv);
}  // namespace geo
}  // namespace MegBA
