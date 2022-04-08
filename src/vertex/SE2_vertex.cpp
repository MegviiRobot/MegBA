/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "vertex/SE2_vertex.h"
#include "vertex/base_vertex.h"

namespace MegAutoBA {
template <typename T>
SE2<T> inverse_SE2_CUDA(const SE2<T> &);

template <typename T>
SE2<T> SE2<T>::inverse() {
  switch (R_.angle().getDevice()) {
    case Device::CUDA:
      return inverse_SE2_CUDA(*this);
    default:
      assert("Not Implemented");
  }
}

template <typename T>
SE2<T> SE2<T>::operator*(const SE2<T> &se2) {
  return SE2<T>();
}

template <typename T>
SE2<T> SE2<T>::operator*(const Eigen::Vector2<JetVector<T>> &t) {
  return SE2<T>();
}

template class SE2<float>;

template class SE2<double>;
}  // namespace MegAutoBA