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
  comms.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i)
    devs[i] = i;
  ncclCommInitAll(comms.data(), MemoryPool::getWorldSize(), devs.data());
}

const std::vector<ncclComm_t> &HandleManager::getNcclComm() { return comms; }

void HandleManager::destroyNcclComm() {
  for (auto comm : comms) {
    ncclCommDestroy(comm);
  }
}

void HandleManager::createCublasHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  assert(cublasHandle.empty());
  cublasHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cublasCreate_v2(&cublasHandle[i]);
  }
}

const std::vector<cublasHandle_t> &HandleManager::getCublasHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  assert(!cublasHandle.empty());
  return cublasHandle;
}

void HandleManager::destroyCublasHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  for (int i = 0; i < cublasHandle.size(); ++i) {
    cudaSetDevice(i);
    cublasDestroy_v2(cublasHandle[i]);
  }
  cublasHandle.clear();
}

void HandleManager::createCusparseHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  assert(cusparseHandle.empty());
  cusparseHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cusparseCreate(&cusparseHandle[i]);
  }
}

const std::vector<cusparseHandle_t> &HandleManager::getCusparseHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  assert(!cusparseHandle.empty());
  return cusparseHandle;
}

void HandleManager::destroyCusparseHandle() {
  std::unique_lock<std::mutex> lock{mutex};
  for (int i = 0; i < cusparseHandle.size(); ++i) {
    cudaSetDevice(i);
    cusparseDestroy(cusparseHandle[i]);
  }
  cusparseHandle.clear();
}

std::vector<ncclComm_t> HandleManager::comms{};
std::vector<cublasHandle_t> HandleManager::cublasHandle{};
std::vector<cusparseHandle_t> HandleManager::cusparseHandle{};
std::mutex HandleManager::mutex{};
}  // namespace MegBA
