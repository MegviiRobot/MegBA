/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "math_function_Jet_Vector_CPU.h"
#include "math_function_Jet_Vector_CUDA.cuh"
#include "JetVector.h"

namespace MegBA {
template <typename T>
inline JetVector<T> operator+(T g, const JetVector<T> &f) {
  return f + g;
}

template <typename T>
inline JetVector<T> operator-(T g, const JetVector<T> &f) {
  return f.scalarMinusThis(g);
}

template <typename T>
inline JetVector<T> operator*(T g, const JetVector<T> &f) {
  return f * g;
}

template <typename T>
inline JetVector<T> operator/(T g, const JetVector<T> &f) {
  return f.scalarDividesThis(g);
}

namespace math {
// In general, f(a + h) ~= f(a) + f'(a) h, via the chain rule.

// abs(x + h) ~= x + h or -(x + h)
template <typename T> inline JetVector<T> abs(const JetVector<T> &f) {
  JetVector<T> out;
  out.Init_as(f);
  switch (f.getDevice()) {
  case CPU: {
    function::absJetVectorCPU(f, out);
    break;
  }
  case CUDA: {
    function::abs_JetVector_CUDA(f, out);
    break;
  }
  } // switch _device
  return out;
}

//        // log(a + h) ~= log(a) + h / a
//        template<typename T>
//        inline
//        JetVector<T> log(JetVector<T> f) {
//            const T a_inverse = T(1.0) / f.a;
//            f.a = log(f.a);
//            f.v = f.v * a_inverse;
//            return f;
//        }
//
//        // exp(a + h) ~= exp(a) + exp(a) h
//        template<typename T>
//        inline
//        JetVector<T> exp(JetVector<T> f) {
//            const T tmp = exp(f.a);
//            f.a = tmp;
//            f.v = tmp * f.v;
//            return f;
//        }
//
// sqrt(a + h) ~= sqrt(a) + h / (2 sqrt(a))
template <typename T> inline JetVector<T> sqrt(JetVector<T> f) {
  JetVector<T> out;
  out.Init_as(f);
  switch (f.getDevice()) {
  case CPU: {
    function::sqrtJetVectorCPU(f, out);
    break;
  }
  case CUDA: {
    function::sqrt_JetVector_CUDA(f, out);
    break;
  }
  } // switch _device
  return out;
}

// cos(a + h) ~= cos(a) - sin(a) h
template <typename T> inline JetVector<T> cos(JetVector<T> f) {
  JetVector<T> out;
  out.Init_as(f);
  switch (f.getDevice()) {
  case CPU: {
    function::cosJetVectorCPU(f, out);
    break;
  }
  case CUDA: {
    function::cos_JetVector_CUDA(f, out);
    break;
  }
  } // switch _device
  return out;
}
//
//        // acos(a + h) ~= acos(a) - 1 / sqrt(1 - a^2) h
//        template<typename T>
//        inline
//        JetVector<T> acos(JetVector<T> f) {
//            const T tmp = -T(1.0) / sqrt(T(1.0) - f.a * f.a);
//            f.a = acos(f.a);
//            f.v = tmp * f.v;
//            return f;
//        }
//
// sin(a + h) ~= sin(a) + cos(a) h
template <typename T> inline JetVector<T> sin(JetVector<T> f) {
  JetVector<T> out;
  out.Init_as(f);
  switch (f.getDevice()) {
  case CPU: {
    function::sinJetVectorCPU(f, out);
    break;
  }
  case CUDA: {
    function::sin_JetVector_CUDA(f, out);
    break;
  }
  } // switch _device
  return out;
}
//
//        // asin(a + h) ~= asin(a) + 1 / sqrt(1 - a^2) h
//        template<typename T>
//        inline
//        JetVector<T> asin(JetVector<T> f) {
//            const T tmp = T(1.0) / sqrt(T(1.0) - f.a * f.a);
//            f.a = asin(f.a);
//            f.v = tmp * f.v;
//            return f;
//        }
//
//        // tan(a + h) ~= tan(a) + (1 + tan(a)^2) h
//        template<typename T>
//        inline
//        JetVector<T> tan(JetVector<T> f) {
//            const T tan_a = tan(f.a);
//            const T tmp = T(1.0) + tan_a * tan_a;
//            f.a = tan_a;
//            f.v = tmp * f.v;
//            return f;
//        }
//
//        // atan(a + h) ~= atan(a) + 1 / (1 + a^2) h
//        template<typename T>
//        inline
//        JetVector<T> atan(JetVector<T> f) {
//            const T tmp = T(1.0) / (T(1.0) + f.a * f.a);
//            f.a = atan(f.a);
//            f.v = tmp * f.v;
//            return f;
//        }
//
//        // sinh(a + h) ~= sinh(a) + cosh(a) h
//        template<typename T>
//        inline
//        JetVector<T> sinh(JetVector<T> f) {
//            f.v = cosh(f.a) * f.v;
//            f.a = sinh(f.a);
//            return f;
//        }
//
//        // cosh(a + h) ~= cosh(a) + sinh(a) h
//        template<typename T>
//        inline
//        JetVector<T> cosh(JetVector<T> f) {
//            f.v = sinh(f.a) * f.v;
//            f.a = cosh(f.a);
//            return f;
//        }
//
//        // tanh(a + h) ~= tanh(a) + (1 - tanh(a)^2) h
//        template<typename T>
//        inline
//        JetVector<T> tanh(JetVector<T> f) {
//            const T tanh_a = tanh(f.a);
//            const T tmp = T(1.0) - tanh_a * tanh_a;
//            f.a = tanh_a;
//            f.v = tmp * f.v;
//            return f;
//        }
//
//        // The floor function should be used with extreme care as this operation will
//        // result in a zero derivative which provides no information to the solver.
//        //
//        // floor(a + h) ~= floor(a) + 0
//        template<typename T>
//        inline
//        JetVector<T> floor(JetVector<T> f) {
//            f.a = floor(f.a);
//            return f;
//        }
//
//        // The ceil function should be used with extreme care as this operation will
//        // result in a zero derivative which provides no information to the solver.
//        //
//        // ceil(a + h) ~= ceil(a) + 0
//        template<typename T>
//        inline
//        JetVector<T> ceil(JetVector<T> f) {
//            f.a = ceil(f.a);
//            return f;
//        }
} // namespace math
}  // namespace MegBA
