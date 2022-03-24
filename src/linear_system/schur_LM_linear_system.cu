/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/schur_LM_linear_system.h"
#include "resource/handle_manager.h"
#include "resource/memory_pool.h"
#include "wrapper.hpp"

namespace MegBA {
namespace {
void CUDART_CB freeCallback(void *ptr) { free(ptr); }

template <typename T>
__global__ void broadCastCsrColInd(const int *input, const int other_dim,
                                   const int nItem, int *output) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem) return;
  for (int i = 0; i < other_dim; ++i) {
    output[i + tid * other_dim] = i + input[tid] * other_dim;
  }
}
}  // namespace

template <typename T>
void SchurLMLinearSystem<T>::allocateResourceCUDA() {
  const auto worldSize = MemoryPool::getWorldSize();
  std::vector<std::array<int *, 2>> compressedCsrColInd;
  compressedCsrColInd.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&this->deltaXPtrBackup[i], this->getHessianShape() * sizeof(T));
    cudaMalloc(&this->deltaXPtr[i], this->getHessianShape() * sizeof(T));
    cudaMemsetAsync(this->deltaXPtr[i], 0, this->getHessianShape() * sizeof(T));

    cudaMalloc(&this->extractedDiag[i][0],
               this->dim[0] * this->num[0] * sizeof(T));
    cudaMalloc(&this->extractedDiag[i][1],
               this->dim[1] * this->num[1] * sizeof(T));

    std::array<int *, 2> csrRowPtrHost{this->equationContainers[i].csrRowPtr};
    cudaMalloc(&this->equationContainers[i].csrRowPtr[0],
               (this->num[0] * this->dim[0] + 1) * sizeof(int));
    cudaMalloc(&this->equationContainers[i].csrRowPtr[1],
               (this->num[1] * this->dim[1] + 1) * sizeof(int));
    cudaMemcpyAsync(this->equationContainers[i].csrRowPtr[0], csrRowPtrHost[0],
                    (this->num[0] * this->dim[0] + 1) * sizeof(int),
                    cudaMemcpyHostToDevice);
    cudaLaunchHostFunc(nullptr, freeCallback, (void *)csrRowPtrHost[0]);
    cudaMemcpyAsync(this->equationContainers[i].csrRowPtr[1], csrRowPtrHost[1],
                    (this->num[1] * this->dim[1] + 1) * sizeof(int),
                    cudaMemcpyHostToDevice);
    cudaLaunchHostFunc(nullptr, freeCallback, (void *)csrRowPtrHost[1]);

    std::array<int *, 2> csrColIndHost{this->equationContainers[i].csrColInd};
    cudaMalloc(&this->equationContainers[i].csrVal[0],
               this->equationContainers[i].nnz[0] * sizeof(T));  // hpl
    cudaMalloc(&this->equationContainers[i].csrColInd[0],
               this->equationContainers[i].nnz[0] * sizeof(int));
    {
      const std::size_t entriesInRows =
          this->equationContainers[i].nnz[0] / this->dim[1];
      dim3 block(std::min(entriesInRows, (std::size_t)512));
      dim3 grid((entriesInRows - 1) / block.x + 1);
      cudaMalloc(&compressedCsrColInd[i][0], entriesInRows * sizeof(int));
      cudaMemcpyAsync(compressedCsrColInd[i][0], csrColIndHost[0],
                      entriesInRows * sizeof(int), cudaMemcpyHostToDevice);
      cudaLaunchHostFunc(nullptr, freeCallback, (void *)csrColIndHost[0]);
      broadCastCsrColInd<T><<<grid, block>>>(
          compressedCsrColInd[i][0], this->dim[1], entriesInRows,
          this->equationContainers[i].csrColInd[0]);
    }

    cudaMalloc(&this->equationContainers[i].csrVal[1],
               this->equationContainers[i].nnz[1] * sizeof(T));  // hlp
    cudaMalloc(&this->equationContainers[i].csrColInd[1],
               this->equationContainers[i].nnz[1] * sizeof(int));
    {
      const std::size_t entriesInRows =
          this->equationContainers[i].nnz[1] / this->dim[0];
      dim3 block(std::min(entriesInRows, (std::size_t)512));
      dim3 grid((entriesInRows - 1) / block.x + 1);
      cudaMalloc(&compressedCsrColInd[i][1], entriesInRows * sizeof(int));
      cudaMemcpyAsync(compressedCsrColInd[i][1], csrColIndHost[1],
                      entriesInRows * sizeof(int), cudaMemcpyHostToDevice);
      cudaLaunchHostFunc(nullptr, freeCallback, (void *)csrColIndHost[1]);
      broadCastCsrColInd<T><<<grid, block>>>(
          compressedCsrColInd[i][1], this->dim[0], entriesInRows,
          this->equationContainers[i].csrColInd[1]);
    }

    cudaMalloc(&this->equationContainers[i].csrVal[2],
               this->equationContainers[i].nnz[2] * sizeof(T));  // hpp

    cudaMalloc(&this->equationContainers[i].csrVal[3],
               this->equationContainers[i].nnz[3] * sizeof(T));  // hll

    cudaMalloc(&this->g[i], this->getHessianShape() * sizeof(T));
    cudaMalloc(&this->gBackup[i], this->getHessianShape() * sizeof(T));
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    cudaFree(compressedCsrColInd[i][0]);
    cudaFree(compressedCsrColInd[i][1]);
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
void SchurLMLinearSystem<T>::processDiag(
    const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const {
  if (lmAlgoStatus.recoverDiag) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = this->equationContainers[i];
      RecoverDiag(this->extractedDiag[i][0], T(1. / lmAlgoStatus.region),
                  this->num[0], this->dim[0], container.csrVal[2]);
      RecoverDiag(this->extractedDiag[i][1], T(1. / lmAlgoStatus.region),
                  this->num[1], this->dim[1], container.csrVal[3]);
    }
  } else {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = this->equationContainers[i];
      extractOldAndApplyNewDiag(T(1. / lmAlgoStatus.region), this->num[0],
                                this->dim[0], container.csrVal[2],
                                this->extractedDiag[i][0]);
      extractOldAndApplyNewDiag(T(1. / lmAlgoStatus.region), this->num[1],
                                this->dim[1], container.csrVal[3],
                                this->extractedDiag[i][1]);
    }
  }
}

template <typename T>
void SchurLMLinearSystem<T>::backup() const {
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
void SchurLMLinearSystem<T>::rollback() const {
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
void SchurLMLinearSystem<T>::applyUpdate(T *xPtr) const {
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  const T one = 1.;
  cudaSetDevice(0);
  Wrapper::cublasGaxpy::call(cublasHandle[0], this->getHessianShape(), &one,
                             this->deltaXPtr[0], 1, xPtr, 1);
}

template struct SchurLMLinearSystem<double>;
template struct SchurLMLinearSystem<float>;
}  // namespace MegBA
