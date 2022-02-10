/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <iostream>
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
    _valueHostVec = f._valueHostVec;
    _gradHostVec = f._gradHostVec;
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
    _valueHostVec = std::move(f._valueHostVec);
    _gradHostVec = std::move(f._gradHostVec);
    break;
  case Device::CUDA:
    _valueDevicePtr = std::move(f._valueDevicePtr);
    _gradDevicePtr = std::move(f._gradDevicePtr);
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
      _valueHostVec.resize(initTemplate._valueHostVec.size());
      _gradHostVec.resize(initTemplate._gradHostVec.size());
      for (unsigned int i = 0; i < _N; ++i)
        _gradHostVec[i].resize(initTemplate._gradHostVec[i].size());
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
  return _valueHostVec.empty() && _valueDevicePtr.empty();
}

template <typename T> void JetVector<T>::set_Grad_Shape(unsigned int N) {
  if (_N != 0)
    throw std::runtime_error("Can not set Grad Shape on a working JetVector, "
                             "use 'Clear()' method first.");
  _N = N;
  _gradHostVec.resize(N);
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
      _valueHostVec.clear();
      _gradHostVec.clear();
      break;
    case Device::CUDA:
      cudaStreamSynchronize(nullptr);
      std::vector<void *> ptrs{_gradDevicePtr.begin(), _gradDevicePtr.end()};
      MemoryPool::deallocateJetVector(&ptrs);
      _valueDevicePtr.clear();
      _gradDevicePtr.clear();
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

  _valueHostVec.push_back(a);
  for (int i = 0; i < _N; ++i)
    _gradHostVec[i].push_back(i == n ? 1 : 0);
  _nItem++;
}

template <typename T> void JetVector<T>::appendJet(T a) {
  assert(_gradPosition >= 0 || _N == 0);
  _valueHostVec.push_back(a);
  _nItem++;
}

template <typename T>
JetVector<T> JetVector<T>::operator+(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, +, g);
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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
  Check::shapeThrow(*this, g);
  Check::deviceThrow(*this, g);
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

template <typename T>
std::ostream &operator<<(std::ostream &s, const MegBA::JetVector<T> &z) {
  switch (z.getDevice()) {
    case MegBA::Device::CPU: {
      s << "[Res: "
        << "[ ";
      for (auto &data : z.getCPURes())
        s << data << ", ";
      s << "]," << std::endl;
      for (unsigned int i = 0; i < z.getGradShape(); ++i) {
        s << "Grad[" << i << "]: "
          << "[ ";
        for (auto &data : z.getCPUGrad()[i])
          s << data << ", ";
        s << "]," << std::endl;
      }
      s << "_device: " << std::to_string(z.getDevice()) << "]";
      break;
    }
    case MegBA::Device::CUDA: {
      return ostreamCUDA(s, z);
    }
  }
  return s;
}
