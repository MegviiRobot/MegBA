/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <cassert>
#include "Common.h"
#include "JetVector.inl"
#include "vector"
#include "Jet_Vector_math.inl"
#include <resource/MemoryPool.h>
#include <functional>
#include <resource/Manager.h>

namespace MegBA {
template <typename T> class JetVector {
  // cuda functions
  void initAsCUDA(const JetVector<T> &f);
  void CPU2CUDA(const JetVector<T> &f);
  void CUDA2CUDA(const JetVector<T> &f);
  void CUDA2CPU(const JetVector<T> &f);

  unsigned int _N = 0;
  unsigned int _nElm = 0;
  device_t _device = CPU_t;
  std::vector<std::vector<T>> _hvData{};
  std::vector<T *> _dvPtr{};
  std::vector<T> _haData{};
  std::vector<T *> _daPtr{};
  int _gradPosition = -1;
  bool _pureScalarFlag = false;
  T _pureScalar = 0;

public:
  JetVector() = default;

  explicit JetVector(T scalar) : _pureScalarFlag(true), _pureScalar(scalar){};

  JetVector(const JetVector<T> &f)
      : _N(f._N), _nElm(f._nElm), _device(f._device), _haData(f._haData),
        _hvData(f._hvData) {
    switch (_device) {
    case CPU_t: {
      break;
    }
    case CUDA_t: {
      CUDA2CUDA(f);
      break;
    }
    }
  };

  JetVector(JetVector<T> &&f) noexcept
      : _N(std::move(f._N)), _nElm(std::move(f._nElm)),
        _device(std::move(f._device)), _hvData(std::move(f._hvData)),
        _dvPtr(std::move(f._dvPtr)), _haData(std::move(f._haData)),
        _daPtr(std::move(f._daPtr)), _gradPosition(std::move(f._gradPosition)),
        _pureScalarFlag(std::move(f._pureScalarFlag)),
        _pureScalar(std::move((f._pureScalar))) {
    f._N = 0;
    f._nElm = 0;
    f._gradPosition = -1;
  };

  template <typename F>
  JetVector(const JetVector<T> &init_template, F &&math_func) {
    initAs(init_template);
    math_func(*this);
  }

  ~JetVector() { clear(); };

  void append_Jet(T a, int n);
  void append_Jet(T a);
  void clear();

  static const JetVector<T> &getInitTemplate(const JetVector<T> &f,
                                             const JetVector<T> &g) {
    return f._N > g._N ? f : g;
  };

  void initAs(const JetVector<T> &initTemplate);
  JetVector<T> &to(device_t device);
  JetVector<T> &CPU();
  JetVector<T> &CUDA();
  bool IsEmpty();
  void set_Grad_Shape(unsigned int N);
  void erase(std::size_t idx) {
    assert(_device == CPU_t || _gradPosition != -1 || _N == 0);
    _haData.erase(_haData.begin() + idx);
    _nElm--;
  };
  const unsigned int &getGradShape() const { return _N; };
  const unsigned int &getElmNum() const { return _nElm; };
  std::size_t getElmNum(int rank) const { return MemoryPool::getElmNum(rank); };
  int getGradPosition() const { return _gradPosition; };
  const device_t &getDevice() const { return _device; };

  const std::vector<std::vector<T>> &getCPUGrad() const { return _hvData; };
  std::vector<std::vector<T>> &getCPUGrad() { return _hvData; };

  const std::vector<T> &getCPURes() const { return _haData; };
  std::vector<T> &getCPURes() { return _haData; };

  // TODO: change to vector
  const std::vector<T *> &getCUDAGradPtr() const { return _dvPtr; };
  const std::vector<T *> &getCUDAResPtr() const { return _daPtr; };

  // TODO: input a array vector
  void bindDaPtr(T *da_ptr) {
    _daPtr.resize(MemoryPool::getWorldSize());
    _daPtr[0] = da_ptr;
  };

  void bindDaPtr(std::vector<T *> &&da_ptr) { _daPtr = std::move(da_ptr); };

  void setGradPosition(int gradPosition);

  JetVector<T> &operator=(const JetVector<T> &f);

  JetVector<T> &operator=(JetVector<T> &&f) noexcept;

  JetVector<T> operator+(const JetVector<T> &g) const;

  JetVector<T> operator-(const JetVector<T> &g) const;

  JetVector<T> operator*(const JetVector<T> &g) const;

  JetVector<T> operator/(const JetVector<T> &g) const;

  JetVector<T> &operator+=(const JetVector<T> &g);

  JetVector<T> &operator-=(const JetVector<T> &g);

  JetVector<T> &operator*=(const JetVector<T> &g);

  JetVector<T> &operator/=(const JetVector<T> &g);

  JetVector<T> operator-() const;

  JetVector<T> operator+(T g) const;

  JetVector<T> operator-(T g) const;

  JetVector<T> operator*(T g) const;

  JetVector<T> operator/(T g) const;

  JetVector<T> &operator+=(T g);

  JetVector<T> &operator-=(T g);

  JetVector<T> &operator*=(T g);

  JetVector<T> &operator/=(T g);

  JetVector<T> scalarMinusThis(T g) const;
  JetVector<T> scalarDividesThis(T g) const;
};

template <typename T>
std::ostream &ostreamCUDA(std::ostream &s, const JetVector<T> &z);

template <typename T>
inline std::ostream &operator<<(std::ostream &s, const JetVector<T> &z) {
  switch (z.getDevice()) {
  case CPU_t: {
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
  case CUDA_t: {
    return ostreamCUDA(s, z);
  }
  }
  return s;
}
}
