/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/implicit_schur_LM_linear_system.h"
#include "resource/handle_manager.h"
#include "resource/memory_pool.h"
#include "wrapper.hpp"

namespace MegBA {
template <typename T>
void ImplicitSchurLMLinearSystem<T>::allocateResourceCUDA() {
  const auto worldSize = MemoryPool::getWorldSize();
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&this->deltaXPtrBackup[i], this->getHessianShape() * sizeof(T));
    cudaMalloc(&this->deltaXPtr[i], this->getHessianShape() * sizeof(T));
    cudaMemsetAsync(this->deltaXPtr[i], 0, this->getHessianShape() * sizeof(T));

    cudaMalloc(&this->extractedDiag[i][0],
               this->dim[0] * this->num[0] * sizeof(T));
    cudaMalloc(&this->extractedDiag[i][1],
               this->dim[1] * this->num[1] * sizeof(T));

    cudaMalloc(&this->implicitEquationContainers[i].csrVal[0],
               this->implicitEquationContainers[i].nnz[0] * sizeof(T));  // hpp

    cudaMalloc(&this->implicitEquationContainers[i].csrVal[1],
               this->implicitEquationContainers[i].nnz[1] * sizeof(T));  // hll

    cudaMalloc(&this->g[i], this->getHessianShape() * sizeof(T));
    cudaMalloc(&this->gBackup[i], this->getHessianShape() * sizeof(T));
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
  }
}

namespace {
template <typename T>
__global__ void RecoverDiagKernel(const T *in, const T a, const int batchSize,
                                  T *out) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many
   * cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize) return;

  out[threadIdx.x + threadIdx.x * blockDim.x + tid * blockDim.x * blockDim.x] =
      (a + 1) * in[threadIdx.x + tid * blockDim.x];
}

template <typename T>
void RecoverDiag(const T *diag, const T a, const int batchSize, const int dim,
                 T *csrVal) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  RecoverDiagKernel<T><<<grid, block>>>(diag, a, batchSize, csrVal);
}

template <typename T>
__global__ void ExtractOldAndApplyNewDiagKernel(const T a, const int batchSize,
                                                T *csrVal, T *diags) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many
   * cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize) return;

  const T diag = csrVal[threadIdx.x + threadIdx.x * blockDim.x +
                        tid * blockDim.x * blockDim.x];
  diags[threadIdx.x + tid * blockDim.x] = diag;
  csrVal[threadIdx.x + threadIdx.x * blockDim.x +
         tid * blockDim.x * blockDim.x] = (a + 1) * diag;
}

template <typename T>
void extractOldAndApplyNewDiag(const T a, const int batchSize, const int dim,
                               T *csrVal, T *diag) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  ExtractOldAndApplyNewDiagKernel<<<grid, block>>>(a, batchSize, csrVal, diag);
}
}  // namespace

template <typename T>
void ImplicitSchurLMLinearSystem<T>::processDiag(
    const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const {
  if (lmAlgoStatus.recoverDiag) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = this->implicitEquationContainers[i];
      RecoverDiag(this->extractedDiag[i][0], T(1. / lmAlgoStatus.region),
                  this->num[0], this->dim[0], container.csrVal[0]);
      RecoverDiag(this->extractedDiag[i][1], T(1. / lmAlgoStatus.region),
                  this->num[1], this->dim[1], container.csrVal[1]);
    }
  } else {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = this->implicitEquationContainers[i];
      extractOldAndApplyNewDiag(T(1. / lmAlgoStatus.region), this->num[0],
                                this->dim[0], container.csrVal[0],
                                this->extractedDiag[i][0]);
      extractOldAndApplyNewDiag(T(1. / lmAlgoStatus.region), this->num[1],
                                this->dim[1], container.csrVal[1],
                                this->extractedDiag[i][1]);
    }
  }
}

template <typename T>
void ImplicitSchurLMLinearSystem<T>::backup() const {
  const int hessianShape = this->getHessianShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaMemcpyAsync(this->deltaXPtrBackup[i], this->deltaXPtr[i],
                    hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(this->gBackup[i], this->g[i], hessianShape * sizeof(T),
                    cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
void ImplicitSchurLMLinearSystem<T>::rollback() const {
  const int hessianShape = this->getHessianShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaMemcpyAsync(this->deltaXPtr[i], this->deltaXPtrBackup[i],
                    hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
    cudaMemcpyAsync(this->g[i], this->gBackup[i], hessianShape * sizeof(T),
                    cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
void ImplicitSchurLMLinearSystem<T>::applyUpdate(T *xPtr) const {
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  const T one = 1.;
  cudaSetDevice(0);
  Wrapper::cublasGaxpy::call(cublasHandle[0], this->getHessianShape(), &one,
                             this->deltaXPtr[0], 1, xPtr, 1);
}

SPECIALIZE_STRUCT(ImplicitSchurLMLinearSystem);
}  // namespace MegBA
