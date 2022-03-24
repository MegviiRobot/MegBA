/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <cassert>
#include <functional>
#include <iostream>
#include <utility>
#include <vector>

#include "common.h"
#include "jet_vector-inl.h"
#include "jet_vector_op-inl.h"
#include "resource/handle_manager.h"
#include "resource/memory_pool.h"

namespace MegBA {
template <typename T>
class JetVector {
  // cuda functions
  void initAsCUDA(const JetVector<T> &f);
  void CPU2CUDA(const JetVector<T> &f);
  void CUDA2CUDA(const JetVector<T> &f);
  void CUDA2CPU(const JetVector<T> &f);

  unsigned int _N{0};
  unsigned int _nItem{0};
  Device _device{Device::CPU};
  std::vector<std::vector<T>> _gradHostVec{};
  std::vector<T *> _gradDevicePtr{};
  std::vector<T> _valueHostVec{};
  std::vector<T *> _valueDevicePtr{};
  int _gradPosition = -1;
  bool _pureScalarFlag = false;
  T _pureScalar = 0;

 public:
  JetVector() = default;

  explicit JetVector(T scalar) : _pureScalarFlag(true), _pureScalar(scalar) {}

  JetVector(const JetVector<T> &f)
      : _N{f._N},
        _nItem{f._nItem},
        _device{f._device},
        _valueHostVec{f._valueHostVec},
        _gradHostVec{f._gradHostVec} {
    switch (_device) {
      case Device::CPU:
        break;
      case Device::CUDA:
        CUDA2CUDA(f);
        break;
    }
  }

  JetVector(JetVector<T> &&f) noexcept
      : _N{f._N},
        _nItem{f._nItem},
        _device{f._device},
        _gradHostVec{std::move(f._gradHostVec)},
        _gradDevicePtr{std::move(f._gradDevicePtr)},
        _valueHostVec{std::move(f._valueHostVec)},
        _valueDevicePtr{std::move(f._valueDevicePtr)},
        _gradPosition{f._gradPosition},
        _pureScalarFlag{f._pureScalarFlag},
        _pureScalar{f._pureScalar} {
    f._N = 0;
    f._nItem = 0;
    f._gradPosition = -1;
  }

  template <typename F>
  JetVector(const JetVector<T> &init_template, F &&math_func) {
    initAs(init_template);
    math_func(this);
  }

  ~JetVector() { clear(); }

  void appendJet(T a, int n);
  void appendJet(T a);
  void clear();
  void clearCUDA();

  static const JetVector<T> &getInitTemplate(const JetVector<T> &f,
                                             const JetVector<T> &g) {
    return f._N > g._N ? f : g;
  }

  void initAs(const JetVector<T> &initTemplate);
  JetVector<T> &to(Device device);
  JetVector<T> &CPU();
  JetVector<T> &CUDA();
  bool IsEmpty();
  void setGradShape(unsigned int N);
  void erase(std::size_t idx) {
    assert(_device == Device::CPU || _gradPosition != -1 || _N == 0);
    _valueHostVec.erase(_valueHostVec.begin() + idx);
    _nItem--;
  }
  const unsigned int &getGradShape() const { return _N; }
  const unsigned int &getItemNum() const { return _nItem; }
  std::size_t getItemNum(int rank) const {
    return MemoryPool::getItemNum(rank);
  }
  int getGradPosition() const { return _gradPosition; }
  const Device &getDevice() const { return _device; }

  const std::vector<std::vector<T>> &getCPUGrad() const { return _gradHostVec; }
  std::vector<std::vector<T>> &getCPUGrad() { return _gradHostVec; }

  const std::vector<T> &getCPURes() const { return _valueHostVec; }
  std::vector<T> &getCPURes() { return _valueHostVec; }

  const std::vector<T *> &getCUDAGradPtr() const { return _gradDevicePtr; }
  const std::vector<T *> &getCUDAResPtr() const { return _valueDevicePtr; }

  void bindValueDevicePtr(std::vector<T *> &&valueDevicePtr) {
    _valueDevicePtr = std::move(valueDevicePtr);
  }

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

  JetVector<T> scalarSubThis(T g) const;
  JetVector<T> scalarDivThis(T g) const;
};
}  // namespace MegBA

template <typename T>
std::ostream &operator<<(std::ostream &s, const MegBA::JetVector<T> &z);

#include "eigen_injector.h"
