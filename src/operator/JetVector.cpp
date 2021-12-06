/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "Common.h"
#include "operator/JetVector.h"
#include "operator/math_function_Jet_Vector_CPU.h"
#include "operator/math_function_Jet_Vector_CUDA.cuh"

#define PURE_SCALAR_OP(f, op, g) \
if ((f).pure_scalar_flag_ || (g).pure_scalar_flag_) { \
    if ((f).pure_scalar_flag_) \
        return (f).pure_scalar_ op (g); \
    else \
        return (f) op (g).pure_scalar_; \
}

namespace MegBA {
template <typename T>
JetVector<T> &JetVector<T>::operator=(const JetVector<T> &f) {
  if (this == &f)
    return *this;
  if (device_ != f.device_ || N_ != f.N_ || nElm_ != f.nElm_) {
    clear();
    N_ = f.N_;
    nElm_ = f.nElm_;
    device_ = f.device_;
    grad_position_ = f.grad_position_;
    pure_scalar_flag_ = f.pure_scalar_flag_;
    pure_scalar_ = f.pure_scalar_;
  }
  switch (device_) {
  case CPU_t:
    ha_data = f.ha_data;
    hv_data = f.hv_data;
    break;
  case CUDA_t:
    CUDA2CUDA(f);
    break;
  }
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator=(JetVector<T> &&f) noexcept {
  clear();
  N_ = f.N_;
  nElm_ = f.nElm_;
  device_ = f.device_;
  grad_position_ = f.grad_position_;
  pure_scalar_flag_ = f.pure_scalar_flag_;
  pure_scalar_ = f.pure_scalar_;
  switch (device_) {
  case CPU_t:
    ha_data = std::move(f.ha_data);
    hv_data = std::move(f.hv_data);
    break;
  case CUDA_t:
    da_ptr_ = std::move(f.da_ptr_);
    dv_ptr_ = std::move(f.dv_ptr_);
    break;
  }
  return *this;
}

template <typename T>
void JetVector<T>::InitAs(const JetVector<T> &Init_Template) {
  if (this == &Init_Template)
    return;

  if (IsEmpty()) {
    device_ = Init_Template.device_;
    nElm_ = Init_Template.nElm_;
    N_ = Init_Template.N_;
    switch (device_) {
    case CPU_t:
      ha_data.resize(Init_Template.ha_data.size());
      hv_data.resize(Init_Template.hv_data.size());
      for (unsigned int i = 0; i < N_; ++i)
        hv_data[i].resize(Init_Template.hv_data[i].size());
      break;
    case CUDA_t:
      cudaInitAs(Init_Template);
      break;
    } // switch device_
  }   // empty
  else
    throw std::runtime_error(
        "You can not init a vector that is not empty."); // !empty
}

template <typename T> JetVector<T> &JetVector<T>::to(device_t device) {
  switch (device) {
  case CPU_t:
    CPU();
    break;
  case CUDA_t:
    CUDA();
    break;
  }
  device_ = device;
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::CPU() {
  if (!IsEmpty()) {
    switch (device_) {
    case CPU_t:
      break;
    case CUDA_t:
      // save counter
      auto N = N_;
      auto nElm = nElm_;
      CUDA2CPU(*this);
      clear();
      // reserve counter
      N_ = N;
      nElm_ = nElm;
      break;
    } // switch device_
  }   // !empty
  device_ = CPU_t;
  return *this;
}

template <typename T> bool JetVector<T>::IsEmpty() {
  return ha_data.empty() && da_ptr_.empty();
}

template <typename T> void JetVector<T>::set_Grad_Shape(unsigned int N) {
  if (N_ != 0)
    throw std::runtime_error("Can not set Grad Shape on a working JetVector, use 'Clear()' method first.");
  N_ = N;
  hv_data.resize(N);
}

template <typename T> void JetVector<T>::set_Grad_Position(int grad_position) {
  assert(grad_position_ == -1 && IsEmpty() &&
         "Can not set Grad Position on a working JetVector, use 'clear()' method first.");
  grad_position_ = grad_position;
}

template <typename T> void JetVector<T>::clear() {
  if (!IsEmpty()) {
    switch (device_) {
    case CPU_t:
      ha_data.clear();
      hv_data.clear();
      break;
    case CUDA_t:
      cudaStreamSynchronize(nullptr);
      std::vector<void *> ptrs{dv_ptr_.begin(), dv_ptr_.end()};
      Memory_Pool::deallocate_JetVector(ptrs);
      da_ptr_.clear();
      dv_ptr_.clear();
      break;
    }
  }
  N_ = 0;
  nElm_ = 0;
}

template <typename T> void JetVector<T>::append_Jet(T a, int n) {
  /*
         * This method not support negative index so far. If n < 0, it has the same effect as a pure_scalar_. 'if (std::abs(n) > N_)' guarantees a negative index won't raise error.
   */
  assert(N_ != 0 && "JetVector does not allow insert Jet while the gradient shape is not initialized.");
  assert(device_ != CUDA_t &&
         "You can not insert Jet into a JetVector in using on CUDA, if you want, use 'clear()' first.");

  ha_data.push_back(a);
  for (int i = 0; i < N_; ++i)
    hv_data[i].push_back(i == n ? 1 : 0);
  nElm_++;
}

template <typename T> void JetVector<T>::append_Jet(T a) {
  //        assert(grad_position_ != -1 && "You should set Grad_Position first before call this method.");
  assert(grad_position_ >= 0 || N_ == 0);
  ha_data.push_back(a);
  nElm_++;
}

template <typename T>
JetVector<T> JetVector<T>::operator+(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, +, g);
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_add_Vector_CPU(*this,
                                                                        g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_add_Vector_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T>
JetVector<T> JetVector<T>::operator-(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, -, g);
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_minus_Vector_CPU(
                               *this, g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_minus_Vector_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T>
JetVector<T> JetVector<T>::operator*(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, *, g);
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_multiplies_Vector_CPU(
                               *this, g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_multiplies_Vector_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T>
JetVector<T> JetVector<T>::operator/(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, /, g);
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_divides_Vector_CPU(
                               *this, g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{get_Init_Template(*this, g), [&](auto &out) {
                           return math::function::Vector_divides_Vector_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T>
JetVector<T> &JetVector<T>::operator+=(const JetVector<T> &g) {
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    math::function::Vector_add_Vector_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::Vector_add_Vector_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator-=(const JetVector<T> &g) {
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    math::function::Vector_minus_Vector_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::Vector_minus_Vector_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator*=(const JetVector<T> &g) {
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    math::function::Vector_multiplies_Vector_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::Vector_multiplies_Vector_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator/=(const JetVector<T> &g) {
  CHK::Shape_Throw(*this, g);
  CHK::Device_Throw(*this, g);
  switch (device_) {
  case CPU_t:
    math::function::Vector_divides_Vector_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::Vector_divides_Vector_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T> JetVector<T> JetVector<T>::operator-() const {
  return T(0) - *this;
}

template <typename T> JetVector<T> JetVector<T>::operator+(T g) const {
  switch (device_) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::JetVector_add_Scalar_CPU(
                               *this, g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::JetVector_add_Scalar_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T> JetVector<T> JetVector<T>::operator-(T g) const {
  switch (device_) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::JetVector_minus_Scalar_CPU(
                               *this, g, out);
                         }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::JetVector_minus_Scalar_CUDA(
                               *this, g, out);
                         }};
  } // switch device_
}

template <typename T> JetVector<T> JetVector<T>::operator*(T g) const {
  switch (device_) {
  case CPU_t:
    return JetVector<T>{
        *this, [&](auto &out) {
          return math::function::JetVector_multiplies_Scalar_CPU(*this, g,
                                                                  out);
        }};
  case CUDA_t:
    return JetVector<T>{
        *this, [&](auto &out) {
          return math::function::JetVector_multiplies_Scalar_CUDA(*this, g,
                                                                   out);
        }};
  } // switch device_
}

template <typename T> JetVector<T> JetVector<T>::operator/(T g) const {
  return *this * T(T(1) / g);
}

template <typename T> JetVector<T> &JetVector<T>::operator+=(T g) {
  switch (device_) {
  case CPU_t:
    math::function::JetVector_add_Scalar_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::JetVector_add_Scalar_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator-=(T g) {
  switch (device_) {
  case CPU_t:
    math::function::JetVector_minus_Scalar_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::JetVector_minus_Scalar_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator*=(T g) {
  switch (device_) {
  case CPU_t:
    math::function::JetVector_multiplies_Scalar_CPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::JetVector_multiplies_Scalar_CUDA(*this, g, *this);
    break;
  } // switch device_
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator/=(T g) {
  g = T(T(1.) / g);
  return (*this *= g);
}

template <typename T> JetVector<T> JetVector<T>::Scalar_minus_this(T f) const {
  switch (device_) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::Scalar_minus_JetVector_CPU(
                               f, *this, out);
                         }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::Scalar_minus_JetVector_CUDA(
                               f, *this, out);
                         }};
  } // switch device_
}

template <typename T>
JetVector<T> JetVector<T>::Scalar_divides_this(T f) const {
  switch (device_) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                           return math::function::Scalar_divides_JetVector_CPU(
                               f, *this, out);
                         }};
  case CUDA_t:
    return JetVector<T>{
        *this, [&](auto &out) {
          return math::function::Scalar_divides_JetVector_CUDA(f, *this, out);
        }};
  } // switch device_
}

template class JetVector<float>;
template class JetVector<double>;
}  // namespace MegBA