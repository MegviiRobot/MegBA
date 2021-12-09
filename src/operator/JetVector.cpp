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
if ((f)._pureScalarFlag || (g)._pureScalarFlag) { \
    if ((f)._pureScalarFlag) \
        return (f)._pureScalar op (g); \
    else \
        return (f) op (g)._pureScalar; \
}

namespace MegBA {
template <typename T>
JetVector<T> &JetVector<T>::operator=(const JetVector<T> &f) {
  if (this == &f)
    return *this;
  if (_device != f._device || _N != f._N || _nElm != f._nElm) {
    clear();
    _N = f._N;
    _nElm = f._nElm;
    _device = f._device;
    _gradPosition = f._gradPosition;
    _pureScalarFlag = f._pureScalarFlag;
    _pureScalar = f._pureScalar;
  }
  switch (_device) {
  case CPU_t:
    _haData = f._haData;
    _hvData = f._hvData;
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
  _N = f._N;
  _nElm = f._nElm;
  _device = f._device;
  _gradPosition = f._gradPosition;
  _pureScalarFlag = f._pureScalarFlag;
  _pureScalar = f._pureScalar;
  switch (_device) {
  case CPU_t:
    _haData = std::move(f._haData);
    _hvData = std::move(f._hvData);
    break;
  case CUDA_t:
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
    _nElm = initTemplate._nElm;
    _N = initTemplate._N;
    switch (_device) {
    case CPU_t:
      _haData.resize(initTemplate._haData.size());
      _hvData.resize(initTemplate._hvData.size());
      for (unsigned int i = 0; i < _N; ++i)
        _hvData[i].resize(initTemplate._hvData[i].size());
      break;
    case CUDA_t:
      initAsCUDA(initTemplate);
      break;
    } // switch _device
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
  _device = device;
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::CPU() {
  if (!IsEmpty()) {
    switch (_device) {
    case CPU_t:
      break;
    case CUDA_t:
      // save counter
      auto N = _N;
      auto nElm = _nElm;
      CUDA2CPU(*this);
      clear();
      // reserve counter
      _N = N;
      _nElm = nElm;
      break;
    } // switch _device
  }   // !empty
  _device = CPU_t;
  return *this;
}

template <typename T> bool JetVector<T>::IsEmpty() {
  return _haData.empty() && _daPtr.empty();
}

template <typename T> void JetVector<T>::set_Grad_Shape(unsigned int N) {
  if (_N != 0)
    throw std::runtime_error("Can not set Grad Shape on a working JetVector, use 'Clear()' method first.");
  _N = N;
  _hvData.resize(N);
}

template <typename T> void JetVector<T>::setGradPosition(int gradPosition) {
  assert(_gradPosition == -1 && IsEmpty() &&
         "Can not set Grad Position on a working JetVector, use 'clear()' method first.");
  _gradPosition = gradPosition;
}

template <typename T> void JetVector<T>::clear() {
  if (!IsEmpty()) {
    switch (_device) {
    case CPU_t:
      _haData.clear();
      _hvData.clear();
      break;
    case CUDA_t:
      cudaStreamSynchronize(nullptr);
      std::vector<void *> ptrs{_dvPtr.begin(), _dvPtr.end()};
      MemoryPool::deallocateJetVector(ptrs);
      _daPtr.clear();
      _dvPtr.clear();
      break;
    }
  }
  _N = 0;
  _nElm = 0;
}

template <typename T> void JetVector<T>::append_Jet(T a, int n) {
  /*
         * This method not support negative index so far. If n < 0, it has the same effect as a _pureScalar. 'if (std::abs(n) > _N)' guarantees a negative index won't raise error.
   */
  assert(_N != 0 && "JetVector does not allow insert Jet while the gradient shape is not initialized.");
  assert(_device != CUDA_t &&
         "You can not insert Jet into a JetVector in using on CUDA, if you want, use 'clear()' first.");

  _haData.push_back(a);
  for (int i = 0; i < _N; ++i)
    _hvData[i].push_back(i == n ? 1 : 0);
  _nElm++;
}

template <typename T> void JetVector<T>::append_Jet(T a) {
  assert(_gradPosition >= 0 || _N == 0);
  _haData.push_back(a);
  _nElm++;
}

template <typename T>
JetVector<T> JetVector<T>::operator+(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, +, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorAddVectorCPU(*this, g,
                                                                    out);
                        }};
  case CUDA_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorAddVectorCUDA(*this, g,
                                                                     out);
                        }};
  } // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator-(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, -, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorMinusVectorCPU(*this, g,
                                                                      out);
                        }};
  case CUDA_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorMinusVectorCUDA(*this, g,
                                                                       out);
                        }};
  } // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator*(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, *, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorMultipliesVectorCPU(
                              *this, g, out);
                        }};
  case CUDA_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorMultipliesVectorCUDA(
                              *this, g, out);
                        }};
  } // switch _device
}

template <typename T>
JetVector<T> JetVector<T>::operator/(const JetVector<T> &g) const {
  PURE_SCALAR_OP(*this, /, g);
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorDividesVectorCPU(*this,
                                                                        g, out);
                        }};
  case CUDA_t:
    return JetVector<T>{getInitTemplate(*this, g), [&](auto &out) {
                          return math::function::vectorDividesVectorCUDA(
                              *this, g, out);
                        }};
  } // switch _device
}

template <typename T>
JetVector<T> &JetVector<T>::operator+=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    math::function::vectorAddVectorCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::vectorAddVectorCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator-=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    math::function::vectorMinusVectorCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::vectorMinusVectorCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator*=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    math::function::vectorMultipliesVectorCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::vectorMultipliesVectorCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T>
JetVector<T> &JetVector<T>::operator/=(const JetVector<T> &g) {
  CHK::shapeThrow(*this, g);
  CHK::deviceThrow(*this, g);
  switch (_device) {
  case CPU_t:
    math::function::vectorDividesVectorCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::vectorDividesVectorCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T> JetVector<T> JetVector<T>::operator-() const {
  return T(0) - *this;
}

template <typename T> JetVector<T> JetVector<T>::operator+(T g) const {
  switch (_device) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorAddScalarCPU(*this, g,
                                                                       out);
                        }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorAddScalarCUDA(*this,
                                                                        g, out);
                        }};
  } // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator-(T g) const {
  switch (_device) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorMinusScalarCPU(
                              *this, g, out);
                        }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorMinusScalarCUDA(
                              *this, g, out);
                        }};
  } // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator*(T g) const {
  switch (_device) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorMultipliesScalarCPU(
                              *this, g, out);
                        }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::jetVectorMultipliesScalarCUDA(
                              *this, g, out);
                        }};
  } // switch _device
}

template <typename T> JetVector<T> JetVector<T>::operator/(T g) const {
  return *this * T(T(1) / g);
}

template <typename T> JetVector<T> &JetVector<T>::operator+=(T g) {
  switch (_device) {
  case CPU_t:
    math::function::jetVectorAddScalarCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::jetVectorAddScalarCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator-=(T g) {
  switch (_device) {
  case CPU_t:
    math::function::jetVectorMinusScalarCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::jetVectorMinusScalarCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator*=(T g) {
  switch (_device) {
  case CPU_t:
    math::function::jetVectorMultipliesScalarCPU(*this, g, *this);
    break;
  case CUDA_t:
    math::function::jetVectorMultipliesScalarCUDA(*this, g, *this);
    break;
  } // switch _device
  return *this;
}

template <typename T> JetVector<T> &JetVector<T>::operator/=(T g) {
  g = T(T(1.) / g);
  return (*this *= g);
}

template <typename T> JetVector<T> JetVector<T>::scalarMinusThis(T f) const {
  switch (_device) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::scalarMinusJetVectorCPU(
                              f, *this, out);
                        }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::scalarMinusJetVectorCUDA(
                              f, *this, out);
                        }};
  } // switch _device
}

template <typename T> JetVector<T> JetVector<T>::scalarDividesThis(T f) const {
  switch (_device) {
  case CPU_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::scalarDividesJetVectorCPU(
                              f, *this, out);
                        }};
  case CUDA_t:
    return JetVector<T>{*this, [&](auto &out) {
                          return math::function::scalarDividesJetVectorCUDA(
                              f, *this, out);
                        }};
  } // switch _device
}

template class JetVector<float>;
template class JetVector<double>;
}  // namespace MegBA