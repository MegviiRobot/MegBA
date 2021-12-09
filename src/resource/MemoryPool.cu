/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "resource/MemoryPool.h"
#include <unordered_map>
#include <stack>
#include "resource/Manager.h"

namespace MegBA {
namespace {
union Ptr {
  explicit Ptr(void *address) : address(address) {}
  void *address;
#if __SIZEOF_POINTER__ == 8
  std::uint64_t number;
#elif __SIZEOF_POINTER__ == 4
  std::uint32_t number;
#elif __SIZEOF_POINTER__ == 2
  std::uint16_t number;
#endif
};

std::vector<std::stack<std::pair<void *, std::size_t>>> ptrRecorder{};
std::vector<std::stack<std::pair<void *, std::size_t>>> overflowedPtrRecorder{};

std::vector<std::size_t> memOffsetCounter{};

std::vector<std::size_t> memOverflowedCounter{};

std::vector<std::size_t> memOverflowedPeak{};

}  // namespace

    void MemoryPool::resetPool(int N, std::size_t nElm, std::int8_t sizeofType,
                            int worldSize) {
  // TODO(Jie Ren): maybe destroy only once
  std::unique_lock<std::mutex> lock{_mutex};
  _N = N;
  _nElm = nElm;
  _sizeofType = sizeofType;
  _worldSize = worldSize;
  HandleManager::destroy_ncclComm();
  HandleManager::create_ncclComm();
  HandleManager::destroy_cublasHandle();
  HandleManager::destroy_cusparseHandle();
  HandleManager::create_cublasHandle();
  HandleManager::create_cusparseHandle();
}

void MemoryPool::allocateJetVector(std::vector<void *> &daPtr,
                                    std::vector<void *> &dvPtr,
                                    std::size_t N, std::size_t nElm,
                                    std::int8_t sizeofType) {
  std::unique_lock<std::mutex> lock{_mutex};
  daPtr.clear();
  daPtr.resize(_worldSize);
  dvPtr.clear();
  dvPtr.resize(_worldSize);
  assert((N == _N || N == 0) && nElm == _nElm && sizeofType == _sizeofType);
  for (auto offset : memOffsetCounter)
    if (offset != 0)
      throw std::runtime_error("memory leak");
  if (_ptr.empty()) {
    for (int i = 0; i < _worldSize; ++i) {
      const auto nElm = getElmNum(i);
      cudaSetDevice(i);
      Ptr ptr{nullptr};
      cudaMalloc(&ptr.address, (_N + 1) * nElm * _sizeofType);
      dvPtr[i] = ptr.address;
      ptr.number += _N * nElm * _sizeofType;
      daPtr[i] = ptr.address;
    }
  } else {
    std::vector<void *> back = std::move(_ptr.back());
    _ptr.pop_back();
    for (int i = 0; i < _worldSize; ++i) {
      const auto nElm = getElmNum(i);
      cudaSetDevice(i);
      Ptr ptr{back[i]};
      dvPtr[i] = ptr.address;
      ptr.number += _N * nElm * _sizeofType;
      daPtr[i] = ptr.address;
    }
  }
  _ptrInUseCounter++;
}

void MemoryPool::deallocateJetVector(std::vector<void *> &ptr) {
  std::unique_lock<std::mutex> lock{_mutex};
  _ptr.push_back(std::move(ptr));
  _ptrInUseCounter--;
}

void MemoryPool::allocateNormal(void **ptr, std::size_t size, int rank) {
  size += size % 8;
  std::unique_lock<std::mutex> lock{_mutex};
  Ptr ptrHelper{nullptr};

  if (memOffsetCounter.empty()) {
    memOffsetCounter.resize(_worldSize);
    ptrRecorder.resize(_worldSize);
    std::fill(memOffsetCounter.begin(), memOffsetCounter.end(), 0);
  }

  bool use_overflowed_stack{_poolSize[rank] < (memOffsetCounter[rank] + size)};
  if (use_overflowed_stack) {
    if (overflowedPtrRecorder.empty()) {
      overflowedPtrRecorder.resize(_worldSize);
      memOverflowedCounter.resize(_worldSize);
      memOverflowedPeak.resize(_worldSize);
      std::fill(memOverflowedCounter.begin(), memOverflowedCounter.end(), 0);
      std::fill(memOverflowedPeak.begin(), memOverflowedPeak.end(), 0);
    }

    memOverflowedPeak[rank] =
        std::max(memOverflowedPeak[rank],
                 memOffsetCounter[rank] + size - _poolSize[rank]);
    cudaSetDevice(rank);
    cudaMalloc(&ptrHelper.address, size);
    overflowedPtrRecorder[rank].emplace(ptrHelper.address, size);
    memOverflowedCounter[rank] += size;
  } else {
    ptrHelper.address = _headPtr[rank];
    ptrHelper.number += memOffsetCounter[rank];
    memOffsetCounter[rank] += size;
  }
  *ptr = ptrHelper.address;
  if (!use_overflowed_stack) {
    ptrRecorder[rank].emplace(ptrHelper.address, size);
  }
}

void MemoryPool::deallocateNormal(void *ptr, int rank) {
  std::unique_lock<std::mutex> lock{_mutex};
  std::pair<void *, std::size_t> back;
  if (ptrRecorder[rank].top().first == ptr) {
    back = std::move(ptrRecorder[rank].top());
    ptrRecorder[rank].pop();
    memOffsetCounter[rank] -= back.second;
  } else {
    if (!overflowedPtrRecorder[rank].empty() &&
        overflowedPtrRecorder[rank].top().first == ptr) {
      back = std::move(overflowedPtrRecorder[rank].top());
      overflowedPtrRecorder[rank].pop();
      cudaSetDevice(rank);
      cudaFree(back.first);
      memOverflowedCounter[rank] -= back.second;
    } else {
      throw std::runtime_error("not using a stack style malloc-free");
    }
  }
}

void MemoryPool::redistribute() {
  if (_poolSize.empty()) {
    _poolSize.resize(_worldSize);
    _headPtr.resize(_worldSize);
    for (int i = 0; i < _worldSize; ++i) {
      cudaSetDevice(i);
      const auto nElm = getElmNum(i);
      for (const auto &v : _ptr) {
        cudaFree(v[i]);
      }
      _poolSize[i] = (_N + 1) * nElm * _sizeofType * _ptr.size();
      cudaMalloc(&_headPtr[i], _poolSize[i]);
      int64_t offset{0};
      for (auto &item : _ptr) {
        Ptr ptr{_headPtr[i]};
        ptr.number += offset;
        offset += (_N + 1) * nElm * _sizeofType;
        item[i] = ptr.address;
      }
    }
  } else {
    bool overflowed{false};
    for (auto peak : memOverflowedPeak)
      overflowed |= peak != 0;
    if (overflowed) {
      for (int i = 0; i < _worldSize; ++i) {
        cudaSetDevice(i);
        const auto nElm = getElmNum(i);
        cudaFree(_headPtr[i]);
        _poolSize[i] += memOverflowedPeak[i];
        cudaMalloc(&_headPtr[i], _poolSize[i]);
        int64_t offset{0};
        for (auto &item : _ptr) {
          Ptr ptr{_headPtr[i]};
          ptr.number += offset;
          offset += (_N + 1) * nElm * _sizeofType;
          item[i] = ptr.address;
        }
      }
    }
  }
}

std::vector<std::vector<void *>> MemoryPool::_ptr{};
std::mutex MemoryPool::_mutex{};
std::vector<std::size_t> MemoryPool::_poolSize{};
std::vector<void *> MemoryPool::_headPtr{};
int MemoryPool::_N{0};
std::size_t MemoryPool::_nElm{0};
std::uint8_t MemoryPool::_sizeofType{0};
int MemoryPool::_worldSize{1};
std::size_t MemoryPool::_ptrInUseCounter{0};
}  // namespace MegBA
