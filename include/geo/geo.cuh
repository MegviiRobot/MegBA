/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "operator/jet_vector.h"

namespace MegBA {
namespace geo {
template <typename T>
using JVD = Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic>;

template <typename T>
using JV3 = Eigen::Matrix<JetVector<T>, 3, 1>;

template <typename T>
using JV4 = Eigen::Matrix<JetVector<T>, 4, 1>;

template <typename T>
using JM33 = Eigen::Matrix<JetVector<T>, 3, 3>;

template <typename T>
using JM22 = Eigen::Matrix<JetVector<T>, 2, 2>;

template <typename T>
JM33<T> AngleAxisToRotationKernelMatrix(const JV3<T> &AxisAngle);

template <typename T>
JM33<T> AngleAxisToRotationKernelMatrix(
    const Eigen::Map<const JVD<T>> &AxisAngle);

template <typename T>
JM22<T> Rotation2DToRotationMatrix(
    const Eigen::Rotation2D<JetVector<T>> &Rotation2D);

template <typename T>
JM33<T> QuaternionToRotationMatrix(const JV4<T> &Q);

template <typename T>
JV4<T> RotationMatrixToQuaternion(const JM33<T> &R);

template <typename T>
JV4<T> &Normalize_(JV4<T> &Q);

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point, const JV3<T> &intrinsic);

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point,
                              const Eigen::Map<const JV3<T>> &intrinsic);

template <typename T>
JetVector<T> RadialDistortion(const JV3<T> &point,
                              const Eigen::Map<const JVD<T>> &intrinsic);

template <typename T>
JVD<T> AnalyticalDerivativesKernelMatrix(
    const Eigen::Map<const JVD<T>> &AxisAngle,
    const Eigen::Map<const JVD<T>> &t,
    const Eigen::Map<const JVD<T>> &intrinsics, const JVD<T> &point_xyz,
    const JVD<T> &obs_uv);

}  // namespace geo
}  // namespace MegBA
