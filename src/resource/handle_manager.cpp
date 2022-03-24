/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <cassert>

#include "resource/handle_manager.h"
#include "resource/memory_pool.h"

namespace MegBA {
void HandleManager::createNCCLComm() {
  comms.resize(MemoryPool::getWorldSize());
  ncclCommInitAll(comms.data(), MemoryPool::getWorldSize(),
                  MemoryPool::getWorld().data());
}

const std::vector<ncclComm_t> &HandleManager::getNCCLComm() { return comms; }

void HandleManager::destroyNCCLComm() {
  for (auto comm : comms) {
    ncclCommDestroy(comm);
  }
}

void HandleManager::createCUBLASHandle() {
  assert(cublasHandle.empty());
  cublasHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cublasCreate_v2(&cublasHandle[i]);
  }
}

const std::vector<cublasHandle_t> &HandleManager::getCUBLASHandle() {
  assert(!cublasHandle.empty());
  return cublasHandle;
}

void HandleManager::destroyCUBLASHandle() {
  for (int i = 0; i < cublasHandle.size(); ++i) {
    cudaSetDevice(i);
    cublasDestroy_v2(cublasHandle[i]);
  }
  cublasHandle.clear();
}

void HandleManager::createCUSPARSEHandle() {
  assert(cusparseHandle.empty());
  cusparseHandle.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cusparseCreate(&cusparseHandle[i]);
  }
}

const std::vector<cusparseHandle_t> &HandleManager::getCUSPARSEHandle() {
  assert(!cusparseHandle.empty());
  return cusparseHandle;
}

void HandleManager::destroyCUSPARSEHandle() {
  for (int i = 0; i < cusparseHandle.size(); ++i) {
    cudaSetDevice(i);
    cusparseDestroy(cusparseHandle[i]);
  }
  cusparseHandle.clear();
}

std::vector<ncclComm_t> HandleManager::comms{};
std::vector<cublasHandle_t> HandleManager::cublasHandle{};
std::vector<cusparseHandle_t> HandleManager::cusparseHandle{};
}  // namespace MegBA
