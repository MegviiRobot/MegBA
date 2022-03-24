/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "jet_vector.h"
#include "jet_vector_math_impl.cuh"
#include "jet_vector_math_impl.h"

namespace MegBA {
template <typename T>
inline JetVector<T> operator+(T g, const JetVector<T> &f) {
  return f + g;
}

template <typename T>
inline JetVector<T> operator-(T g, const JetVector<T> &f) {
  return f.scalarSubThis(g);
}

template <typename T>
inline JetVector<T> operator*(T g, const JetVector<T> &f) {
  return f * g;
}

template <typename T>
inline JetVector<T> operator/(T g, const JetVector<T> &f) {
  return f.scalarDivThis(g);
}

namespace math {

template <typename T>
inline JetVector<T> abs(const JetVector<T> &f) {
  switch (f.getDevice()) {
    case Device::CPU:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::absJetVectorCPU(f, out);
                          }};
    case Device::CUDA:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::absJetVectorCUDA(f, out);
                          }};
  }  // switch _device
}

template <typename T>
inline JetVector<T> sqrt(const JetVector<T> &f) {
  switch (f.getDevice()) {
    case Device::CPU:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::sqrtJetVectorCPU(f, out);
                          }};
    case Device::CUDA:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::sqrtJetVectorCUDA(f, out);
                          }};
  }  // switch _device
}

template <typename T>
inline JetVector<T> cos(const JetVector<T> &f) {
  switch (f.getDevice()) {
    case Device::CPU:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::cosJetVectorCPU(f, out);
                          }};
    case Device::CUDA:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::cosJetVectorCUDA(f, out);
                          }};
  }  // switch _device
}

template <typename T>
inline JetVector<T> sin(const JetVector<T> &f) {
  switch (f.getDevice()) {
    case Device::CPU:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::sinJetVectorCPU(f, out);
                          }};
    case Device::CUDA:
      return JetVector<T>{f, [&](JetVector<T> *out) {
                            return math::impl::sinJetVectorCUDA(f, out);
                          }};
  }  // switch _device
}
}  // namespace math
}  // namespace MegBA
