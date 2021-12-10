/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "resource/handle_manager.h"
#include <cassert>
#include "resource/memory_pool.h"

namespace MegBA {
void HandleManager::createNcclComm() {
  std::vector<int> devs;
  devs.resize(MemoryPool::getWorldSize());
  _comms.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i)
    devs[i] = i;
  ncclCommInitAll(_comms.data(), MemoryPool::getWorldSize(), devs.data());
}

const std::vector<ncclComm_t> &HandleManager::getNcclComm() { return _comms; }

void HandleManager::destroyNcclComm() {
  for (auto comm : _comms) {
    ncclCommDestroy(comm);
  }
}

void HandleManager::createCublasHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  assert(_cublasHandle.empty());
  _cublasHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cublasCreate_v2(&_cublasHandle[i]);
  }
}

const std::vector<cublasHandle_t> &HandleManager::getCublasHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  assert(!_cublasHandle.empty());
  return _cublasHandle;
}

void HandleManager::destroyCublasHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  for (int i = 0; i < _cublasHandle.size(); ++i) {
    cudaSetDevice(i);
    cublasDestroy_v2(_cublasHandle[i]);
  }
  _cublasHandle.clear();
}

void HandleManager::createCusparseHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  assert(_cusparseHandle.empty());
  _cusparseHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cusparseCreate(&_cusparseHandle[i]);
  }
}

const std::vector<cusparseHandle_t> &HandleManager::getCusparseHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  assert(!_cusparseHandle.empty());
  return _cusparseHandle;
}

void HandleManager::destroyCusparseHandle() {
  std::unique_lock<std::mutex> lock{_mutex};
  for (int i = 0; i < _cusparseHandle.size(); ++i) {
    cudaSetDevice(i);
    cusparseDestroy(_cusparseHandle[i]);
  }
  _cusparseHandle.clear();
}

std::vector<ncclComm_t> HandleManager::_comms{};
std::vector<cublasHandle_t> HandleManager::_cublasHandle{};
std::vector<cusparseHandle_t> HandleManager::_cusparseHandle{};
std::mutex HandleManager::_mutex{};
}  // namespace MegBA
