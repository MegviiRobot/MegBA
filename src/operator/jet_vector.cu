/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "operator/jet_vector.h"
#include <memory>
#include "resource/memory_pool.h"
#include "common.h"

namespace MegBA {
template <typename T> void JetVector<T>::initAsCUDA(const JetVector<T> &f) {
  const auto world_size = MemoryPool::getWorldSize();
  std::vector<void *> da_ptr, dv_ptr;
  MemoryPool::allocateJetVector(&da_ptr, &dv_ptr, _N, _nEle, sizeof(T));
  _dvPtr.clear();
  _daPtr.clear();
  _dvPtr.resize(world_size);
  _daPtr.resize(world_size);
  for (int i = 0; i < world_size; ++i) {
    _dvPtr[i] = reinterpret_cast<T *>(dv_ptr[i]);
    _daPtr[i] = reinterpret_cast<T *>(da_ptr[i]);
  }
}

template <typename T> JetVector<T> &JetVector<T>::CUDA() {
  if (!IsEmpty()) {
    auto N = _N;
    auto nEle = _nEle;
    switch (_device) {
    case Device::CUDA: {
      break;
    }
    case Device::CPU: {
      // save counter
      CPU2CUDA(*this);
      clear();
      break;
    }
    }  // switch _device
    _N = N;
    _nEle = nEle;
  }
  _device = Device::CUDA;
  return *this;
}

template <typename T> void JetVector<T>::CUDA2CPU(const JetVector<T> &f) {
  const auto world_size = MemoryPool::getWorldSize();
  _haData.resize(_nEle);
  _hvData.resize(_N);
  for (auto &v : _hvData)
    v.resize(_nEle);

  std::size_t start_idx{0};
  for (int i = 0; i < world_size; ++i) {
    cudaSetDevice(i);
    std::size_t nEle{getEleNum(i)};
    cudaMemcpyAsync(&_haData[start_idx], f._daPtr[i], nEle * sizeof(T),
                    cudaMemcpyDeviceToHost);
    if (_gradPosition == -1) {
      for (unsigned int j = 0; j < _N; ++j)
        cudaMemcpyAsync(&_hvData[j][start_idx], &f._dvPtr[i][j * nEle],
                        nEle * sizeof(T), cudaMemcpyDeviceToHost);
    }
    start_idx += nEle;
  }
}

template <typename T> void JetVector<T>::CPU2CUDA(const JetVector<T> &f) {
  const auto world_size = MemoryPool::getWorldSize();
  // if _daPtr != nullptr if binded
  if (_daPtr.empty()) {
    if (_pureScalarFlag) {
      _daPtr.resize(world_size);
      std::size_t start_idx{0};
      for (int i = 0; i < world_size; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&_daPtr[i], _nEle * sizeof(T));
        std::size_t nEle{getEleNum(i)};
        cudaMemcpyAsync(_daPtr[i], &f._haData[start_idx], nEle * sizeof(T),
                        cudaMemcpyHostToDevice);
        start_idx += nEle;
      }
      return;
    }
    std::vector<void *> da_ptr{}, dv_ptr{};
    MemoryPool::allocateJetVector(&da_ptr, &dv_ptr, _N, _nEle, sizeof(T));
    // _dvPtr must be nullptr
    _dvPtr.clear();
    _dvPtr.reserve(world_size);
    for (int i = 0; i < world_size; ++i)
      _dvPtr.push_back(reinterpret_cast<T *>(dv_ptr[i]));

    _daPtr.clear();
    _daPtr.reserve(world_size);
    for (int i = 0; i < world_size; ++i)
      _daPtr.push_back(reinterpret_cast<T *>(da_ptr[i]));

    std::size_t start_idx{0};
    for (int i = 0; i < world_size; ++i) {
      cudaSetDevice(i);
      std::size_t nEle{getEleNum(i)};
      cudaMemcpyAsync(_daPtr[i], &f._haData[start_idx], nEle * sizeof(T),
                      cudaMemcpyHostToDevice);
      for (unsigned int j = 0; j < _N; ++j)
        cudaMemcpyAsync(&_dvPtr[i][j * nEle], &f._hvData[j][start_idx],
                        nEle * sizeof(T), cudaMemcpyHostToDevice);
      start_idx += nEle;
    }
  } else {
    std::size_t start_idx{0};
    for (int i = 0; i < world_size; ++i) {
      cudaSetDevice(i);
      std::size_t nEle{getEleNum(i)};
      cudaMemcpyAsync(_daPtr[i], &f._haData[start_idx], nEle * sizeof(T),
                      cudaMemcpyHostToDevice);
      start_idx += nEle;
    }
  }
}

template <typename T> void JetVector<T>::CUDA2CUDA(const JetVector<T> &f) {
  const auto world_size = MemoryPool::getWorldSize();
  if (_daPtr.empty()) {
    std::vector<void *> da_ptr{}, dv_ptr{};
    MemoryPool::allocateJetVector(&da_ptr, &dv_ptr, _N, _nEle, sizeof(T));
    _dvPtr.clear();
    _daPtr.clear();
    _dvPtr.reserve(world_size);
    _daPtr.reserve(world_size);
    for (int i = 0; i < world_size; ++i) {
      _dvPtr.push_back(reinterpret_cast<T *>(dv_ptr[i]));
      _daPtr.push_back(reinterpret_cast<T *>(da_ptr[i]));
    }
  }
  for (int i = 0; i < world_size; ++i) {
    cudaSetDevice(i);
    std::size_t nEle{getEleNum(i)};
    cudaMemcpyAsync(_daPtr[i], f._daPtr[i], nEle * sizeof(T),
                    cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(_dvPtr[i], f._dvPtr[i], _N * nEle * sizeof(T),
                    cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
std::ostream &ostreamCUDA(std::ostream &s, const JetVector<T> &z) {
  auto N = z.getGradShape();
  auto nEle = z.getEleNum();
  std::unique_ptr<T[]> Res{new T[nEle]};
  std::vector<std::unique_ptr<T[]>> Grad;
  Grad.reserve(N);
  for (int i = 0; i < N; ++i)
    Grad.emplace_back(new T[nEle]);
  std::size_t start_idx{0};
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    std::size_t nEle = z.getEleNum(i);
    cudaMemcpyAsync(&Res[start_idx], z.getCUDAResPtr()[i], nEle * sizeof(T),
                    cudaMemcpyDeviceToHost);
    for (unsigned int j = 0; j < N; ++j)
      cudaMemcpyAsync(&Grad[j][start_idx], &z.getCUDAGradPtr()[i][j * nEle],
                      nEle * sizeof(T), cudaMemcpyDeviceToHost);
    start_idx += nEle;
  }
  s << "[Res: "
    << "[ ";
  for (std::size_t i = 0; i < nEle; ++i)
    s << Res[i] << ", ";
  s << "]," << std::endl;
  for (unsigned int i = 0; i < N; ++i) {
    s << "Grad[" << i << "]: "
      << "[ ";
    for (std::size_t j = 0; j < nEle; ++j)
      s << Grad[i][j] << ", ";
    s << "]," << std::endl;
  }
  s << "_device: " << std::to_string(z.getDevice()) << "]";
  return s;
}
template std::ostream &ostreamCUDA<float>(std::ostream &s,
                                          const JetVector<float> &);
template std::ostream &ostreamCUDA<double>(std::ostream &s,
                                           const JetVector<double> &);

template class JetVector<float>;
template class JetVector<double>;
}  // namespace MegBA
