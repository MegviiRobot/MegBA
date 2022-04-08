/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <Eigen/Geometry>

#include "operator/jet_vector.h"
#include "vertex/base_vertex.h"

namespace MegBA {
template <typename T>
class SE2 {
  Eigen::Rotation2D<JetVector<T>> R_{};
  Eigen::Vector2<JetVector<T>> t_{};

  void set_Rotation(const JetVector<T> &R) { R_.angle() = R; }

  void set_Translation(const Eigen::Vector2<JetVector<T>> &t) { t_ = t; }

 public:
  SE2() = default;

  Eigen::Rotation2D<JetVector<T>> &rotation() { return R_; }

  const Eigen::Rotation2D<JetVector<T>> &rotation() const { return R_; }

  Eigen::Vector2<JetVector<T>> &translation() { return t_; }

  const Eigen::Vector2<JetVector<T>> &translation() const { return t_; }

  SE2<T> inverse();

  SE2<T> operator*(const SE2<T> &se2);

  SE2<T> operator*(const Eigen::Vector2<JetVector<T>> &t);
};

template <typename T>
struct SE2_Vertex : public BaseVertex<T> {
  SE2<T> estimation_;
};
}  // namespace MegBA
