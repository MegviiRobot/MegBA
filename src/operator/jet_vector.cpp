/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "common.h"
#include "operator/jet_vector.h"
#include "operator/jet_vector_math_impl.h"
#include "operator/jet_vector_math_impl.cuh"

#define PURE_SCALAR_OP(f, op, g)                  \
if ((f)._pureScalarFlag || (g)._pureScalarFlag) { \
  if ((f)._pureScalarFlag) {                      \
    return (f)._pureScalar op(g);                 \
  } else {                                        \
    return (f)op(g)._pureScalar;                  \
  }                                               \
}

namespace MegBA {
template <typename T>
JetVector<T> &JetVector<T>::operator=(const JetVector<T> &f) {
  if (this == &f)
    return *this;
  if (_device != f._device || _N != f._N || _nItem != f._nItem) {
    clear();
    _N = f._N;
    _nItem = f._nItem;
    _device = f._device;
    _gradPosition = f._gradPosition;
    _pureScalarFlag = f._pureScalarFlag;
    _pureScalar = f._pureScalar;
  }
  switch (_device) {
  case Device::CPU:
    _haData = f._haData;
    _hvData = f._hvData;
    break;
  case Device::CUDA:
    CUDA2CUDA(f);
    break;
  }
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator=(JetVector<T> &&f) noexcept {
  clear();
  _N = f._N;
  _nItem = f._nItem;
  _device = f._device;
  _gradPosition = f._gradPosition;
  _pureScalarFlag = f._pureScalarFlag;
  _pureScalar = f._pureScalar;
  switch (_device) {
  case Device::CPU:
    _haData = std::move(f._haData);
    _hvData = std::move(f._hvData);
    break;
  case Device::CUDA:
    _daPtr = std::move(f._daPtr);
    _dvPtr = std::move(f._dvPtr);
    break;
  }
  return *this;
}

template <typename T>
void JetVector<T>::initAs(const JetVector<T> &initTemplate) {
  if (this == &initTemplate)
    return;

  if (IsEmpty()) {
    _device = initTemplate._device;
    _nItem = initTemplate._nItem;
    _N = initTemplate._N;
    switch (_device) {
    case Device::CPU:
      _haData.resize(initTemplate._haData.size());
      _hvData.resize(initTemplate._hvData.size());
      for (unsigned int i = 0; i < _N; ++i)
        _hvData[i].resize(initTemplate._hvData[i].size());
      break;
    case Device::CUDA:
      initAsCUDA(initTemplate);
      break;
    }  // switch _device
  } else {
    throw std::runtime_error(
        "You can not init a vector that is not empty.");  // !empty
  }
}

template <typename T> JetVector<T> &JetVector<T>::to(Device device) {
  switch (device) {
  case Device::CPU:
    CPU();
    break;
  case Device::CUDA:
    CUDA();
    break;
  }
  _device = device;
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::CPU() {
  if (!IsEmpty()) {
    switch (_device) {
    case Device::CPU:
      break;
    case Device::CUDA:
      // save counter
      auto N = _N;
      auto nItem = _nItem;
      CUDA2CPU(*this);
      clear();
      // reserve counter
      _N = N;
      _nItem = nItem;
      break;
    }  // switch _device
  }  // !empty
  _device = Device::CPU;
  return *this;
}

template <typename T> bool JetVector<T>::IsEmpty() {
  return _haData.empty() && _daPtr.empty();
}

template <typename T> void JetVector<T>::set_Grad_Shape(unsigned int N) {
  if (_N != 0)
    throw std::runtime_error("Can not set Grad Shape on a working JetVector, "
                             "use 'Clear()' method first.");
  _N = N;
  _hvData.resize(N);
}

template <typename T> void JetVector<T>::setGradPosition(int gradPosition) {
  assert(_gradPosition == -1 && IsEmpty() &&
         "Can not set Grad Position on a working JetVector, "
         "use 'clear()' method first.");
  _gradPosition = gradPosition;
}

template <typename T> void JetVector<T>::clear() {
  if (!IsEmpty()) {
    switch (_device) {
    case Device::CPU:
      _haData.clear();
      _hvData.clear();
      break;
    case Device::CUDA:
      cudaStreamSynchronize(nullptr);
      std::vector<void *> ptrs{_dvPtr.begin(), _dvPtr.end()};
      MemoryPool::deallocateJetVector(&ptrs);
      _daPtr.clear();
      _dvPtr.clear();
      break;
    }
  }
  _N = 0;
  _nItem = 0;
}

template <typename T> void JetVector<T>::appendJet(T a, int n) {
  /*
         * This method not support negative index so far. If n < 0, it has the same effect as a _pureScalar. 'if (std::abs(n) > _N)' guarantees a negative index won't raise error.
   */
  assert(_N != 0 && "JetVector does not allow insert Jet "
                    "while the gradient shape is not initialized.");
  assert(_device != Device::CUDA &&
         "You can not insert Jet into a JetVector in using on CUDA, "
         "if you want, use 'clear()' first.");

  _haData.push_back(a);
  for (int i = 0; i < _N; ++i)
    _hvData[i].push_back(i == n ? 1 : 0);
  _nItem++;
}

template <typename T> void JetVector<T>::appendJet(T a) {
  assert(_gradPosition >= 0 || _N == 0);
  _haData.push_back(a);
  _nItem++;
}

template <typename T>
JetVector<T> JetVector<T>::operator+(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, +, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorAddVectorCPU(*this, g,
                                                                    out);
                        }};
  case Device::CUDA:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorAddVectorCUDA(*this, g,
                                                                     out);
                        }};
  }  // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator-(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, -, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorSubVectorCPU(*this, g,
                                                                      out);
                        }};
  case Device::CUDA:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorSubVectorCUDA(*this, g,
                                                                       out);
                        }};
  }  // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator*(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, *, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorMulVectorCPU(
                              *this, g, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorMulVectorCUDA(
                              *this, g, out);
                        }};
  }  // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator/(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, /, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorDivVectorCPU(*this,
                                                                        g, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{getInitTemplate(*this, g), [&](JetVector<T> *out) {
                          return math::impl::vectorDivVectorCUDA(
                              *this, g, out);
                        }};
  }  // switch _device
}

template <typename T>
JetVector<T> &JetVector<T>::operator+=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    math::impl::vectorAddVectorCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::vectorAddVectorCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator-=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    math::impl::vectorSubVectorCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::vectorSubVectorCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator*=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    math::impl::vectorMulVectorCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::vectorMulVectorCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator/=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case Device::CPU:
    math::impl::vectorDivVectorCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::vectorDivVectorCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T> JetVector<T> JetVector<T>::operator-() const {
  return T(0) - *this;
}

template <typename T> JetVector<T> JetVector<T>::operator+(T g) const {
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorAddScalarCPU(*this, g,
                                                                       out);
                        }};
  case Device::CUDA:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorAddScalarCUDA(*this,
                                                                        g, out);
                        }};
  }  // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator-(T g) const {
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorSubScalarCPU(
                              *this, g, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorSubScalarCUDA(
                              *this, g, out);
                        }};
  }  // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator*(T g) const {
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorMulScalarCPU(
                              *this, g, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::jetVectorMulScalarCUDA(
                              *this, g, out);
                        }};
  }  // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator/(T g) const {
  return *this * T(T(1) / g);
}

template <typename T> JetVector<T> &JetVector<T>::operator+=(T g) {
  switch (_device) {
  case Device::CPU:
    math::impl::jetVectorAddScalarCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::jetVectorAddScalarCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator-=(T g) {
  switch (_device) {
  case Device::CPU:
    math::impl::jetVectorSubScalarCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::jetVectorSubScalarCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator*=(T g) {
  switch (_device) {
  case Device::CPU:
    math::impl::jetVectorMulScalarCPU(*this, g, this);
    break;
  case Device::CUDA:
    math::impl::jetVectorMulScalarCUDA(*this, g, this);
    break;
  }  // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator/=(T g) {
  g = T(T(1.) / g);
  return (*this *= g);
}

template <typename T> JetVector<T> JetVector<T>::scalarSubThis(T f) const {
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::scalarSubJetVectorCPU(
                              f, *this, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::scalarSubJetVectorCUDA(
                              f, *this, out);
                        }};
  }  // switch _device
}

template <typename T> JetVector<T> JetVector<T>::scalarDivThis(T f) const {
  switch (_device) {
  case Device::CPU:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::scalarDivJetVectorCPU(
                              f, *this, out);
                        }};
  case Device::CUDA:
    return JetVector<T>{*this, [&](JetVector<T> *out) {
                          return math::impl::scalarDivJetVectorCUDA(
                              f, *this, out);
                        }};
  }  // switch _device
}

template class JetVector<float>;
template class JetVector<double>;
}  // namespace MegBA
