/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <memory>

#include "operator/jet_vector.h"
#include "resource/memory_pool.h"

namespace MegBA {
template <typename T>
void JetVector<T>::initAsCUDA(const JetVector<T> &f) {
  const auto worldSize = MemoryPool::getWorldSize();
  std::vector<void *> valueDevicePtr, gradDeviceptr;
  MemoryPool::allocateJetVector(valueDevicePtr, gradDeviceptr, _N, _nItem,
                                sizeof(T));
  _gradDevicePtr.clear();
  _valueDevicePtr.clear();
  _gradDevicePtr.resize(worldSize);
  _valueDevicePtr.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    _gradDevicePtr[i] = reinterpret_cast<T *>(gradDeviceptr[i]);
    _valueDevicePtr[i] = reinterpret_cast<T *>(valueDevicePtr[i]);
  }
}

template <typename T>
JetVector<T> &JetVector<T>::CUDA() {
  if (!IsEmpty()) {
    auto N = _N;
    auto nItem = _nItem;
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
    _nItem = nItem;
  }
  _device = Device::CUDA;
  return *this;
}

template <typename T>
void JetVector<T>::CUDA2CPU(const JetVector<T> &f) {
  const auto worldSize = MemoryPool::getWorldSize();
  _valueHostVec.resize(_nItem);
  _gradHostVec.resize(_N);
  for (auto &v : _gradHostVec) v.resize(_nItem);

  std::size_t startIdx{0};
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    std::size_t nItem{getItemNum(i)};
    cudaMemcpyAsync(&_valueHostVec[startIdx], f._valueDevicePtr[i],
                    nItem * sizeof(T), cudaMemcpyDeviceToHost);
    if (_gradPosition == -1) {
      for (unsigned int j = 0; j < _N; ++j)
        cudaMemcpyAsync(&_gradHostVec[j][startIdx],
                        &f._gradDevicePtr[i][j * nItem], nItem * sizeof(T),
                        cudaMemcpyDeviceToHost);
    }
    startIdx += nItem;
  }
}

template <typename T>
void JetVector<T>::CPU2CUDA(const JetVector<T> &f) {
  const auto worldSize = MemoryPool::getWorldSize();
  // if _valueDevicePtr != nullptr if binded
  if (_gradPosition == -1 && _N != 0) {
    // gradPosition == -1 and N != 0, it would be a normal JV
    std::vector<void *> valueDevicePtr{}, gradDevicePtr{};
    MemoryPool::allocateJetVector(valueDevicePtr, gradDevicePtr, _N, _nItem,
                                  sizeof(T));
    // _gradDevicePtr must be nullptr
    _gradDevicePtr.clear();
    _gradDevicePtr.reserve(worldSize);
    for (int i = 0; i < worldSize; ++i)
      _gradDevicePtr.push_back(reinterpret_cast<T *>(gradDevicePtr[i]));

    _valueDevicePtr.clear();
    _valueDevicePtr.reserve(worldSize);
    for (int i = 0; i < worldSize; ++i)
      _valueDevicePtr.push_back(reinterpret_cast<T *>(valueDevicePtr[i]));

    std::size_t startIdx{0};
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      std::size_t nItem{getItemNum(i)};
      for (unsigned int j = 0; j < _N; ++j)
        cudaMemcpyAsync(&_gradDevicePtr[i][j * nItem],
                        &f._gradHostVec[j][startIdx], nItem * sizeof(T),
                        cudaMemcpyHostToDevice);
      startIdx += nItem;
    }
  } else if (_N == 0) {
    if (_pureScalarFlag) {
      return;
    } else {
      // grad == 0 and non-pureScalar, it would be PVector
      _valueDevicePtr.resize(worldSize);
      for (int i = 0; i < worldSize; ++i) {
        cudaSetDevice(i);
        cudaMalloc(&_valueDevicePtr[i], getItemNum(i) * sizeof(T));
      }
    }
  }

  std::size_t startIdx{0};
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    std::size_t nItem{getItemNum(i)};
    cudaMemcpyAsync(_valueDevicePtr[i], &f._valueHostVec[startIdx],
                    nItem * sizeof(T), cudaMemcpyHostToDevice);
    startIdx += nItem;
  }
}

template <typename T>
void JetVector<T>::CUDA2CUDA(const JetVector<T> &f) {
  const auto worldSize = MemoryPool::getWorldSize();
  if (_valueDevicePtr.empty()) {
    std::vector<void *> valueDevicePtr{}, gradDevicePtr{};
    MemoryPool::allocateJetVector(valueDevicePtr, gradDevicePtr, _N, _nItem,
                                  sizeof(T));
    _gradDevicePtr.clear();
    _valueDevicePtr.clear();
    _gradDevicePtr.reserve(worldSize);
    _valueDevicePtr.reserve(worldSize);
    for (int i = 0; i < worldSize; ++i) {
      _gradDevicePtr.push_back(reinterpret_cast<T *>(gradDevicePtr[i]));
      _valueDevicePtr.push_back(reinterpret_cast<T *>(valueDevicePtr[i]));
    }
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    std::size_t nItem{getItemNum(i)};
    cudaMemcpyAsync(_valueDevicePtr[i], f._valueDevicePtr[i], nItem * sizeof(T),
                    cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(_gradDevicePtr[i], f._gradDevicePtr[i],
                    _N * nItem * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
void JetVector<T>::clearCUDA() {
  cudaStreamSynchronize(nullptr);
  if (_gradPosition == -1 && _N != 0) {
    std::vector<void *> ptrs{_gradDevicePtr.begin(), _gradDevicePtr.end()};
    MemoryPool::deallocateJetVector(ptrs);
  } else if (_gradPosition == -1 && _N == 0) {
    if (!_pureScalarFlag) {
      const auto &world = MemoryPool::getWorld();
      for (int i = 0; i < world.size(); ++i) {
        cudaSetDevice(world[i]);
        cudaFree(_valueDevicePtr[i]);
      }
    }
  } else {
    ;
  }
  _valueDevicePtr.clear();
  _gradDevicePtr.clear();
}

template <typename T>
std::ostream &ostreamCUDA(std::ostream &s, const JetVector<T> &z) {
  auto N = z.getGradShape();
  auto nItem = z.getItemNum();
  std::unique_ptr<T[]> Res{new T[nItem]};
  std::vector<std::unique_ptr<T[]>> Grad;
  Grad.reserve(N);
  for (int i = 0; i < N; ++i) Grad.emplace_back(new T[nItem]);
  std::size_t startIdx{0};
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    std::size_t nItem = z.getItemNum(i);
    cudaMemcpyAsync(&Res[startIdx], z.getCUDAResPtr()[i], nItem * sizeof(T),
                    cudaMemcpyDeviceToHost);
    for (unsigned int j = 0; j < N; ++j)
      cudaMemcpyAsync(&Grad[j][startIdx], &z.getCUDAGradPtr()[i][j * nItem],
                      nItem * sizeof(T), cudaMemcpyDeviceToHost);
    startIdx += nItem;
  }
  s << "[Res: "
    << "[ ";
  for (std::size_t i = 0; i < nItem; ++i) s << Res[i] << ", ";
  s << "]," << std::endl;
  for (unsigned int i = 0; i < N; ++i) {
    s << "Grad[" << i << "]: "
      << "[ ";
    for (std::size_t j = 0; j < nItem; ++j) s << Grad[i][j] << ", ";
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
