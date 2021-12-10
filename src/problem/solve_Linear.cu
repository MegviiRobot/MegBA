/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include <cublas_v2.h>
#include <cusparse_v2.h>
#include "problem/BaseProblem.h"
#include "edge/base_edge.h"
#include "Wrapper.hpp"
#include "resource/Manager.h"
#include "Macro.h"

#if __CUDA_ARCH__ <= 1120
#define CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#endif

namespace MegBA {
namespace {
template <typename T>
__global__ void weightedPlusKernel(int nElm, const T *x, const T *y, T weight,
                                   T *z) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nElm)
    return;
  z[tid] = x[tid] + weight * y[tid];
}

template <typename T>
__global__ void fillPtr(const T *aData, T *ainvData, const int batchSize,
                        const int hRowsNumPow2, const T **a, T **ainv) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= batchSize)
    return;
  a[tid] = &aData[tid * hRowsNumPow2];
  ainv[tid] = &ainvData[tid * hRowsNumPow2];
}

template <typename T>
void invert(const T *aFlat, int n, const int pointNum, T *cFlat) {
  cublasHandle_t handle = HandleManager::get_cublasHandle()[0];

  const T **a;
  T **ainv;
  cudaMalloc(&a, pointNum * sizeof(T *));
  cudaMalloc(&ainv, pointNum * sizeof(T *));
  dim3 blockDim(std::min(decltype(pointNum)(256), pointNum));
  dim3 gridDim((pointNum - 1) / blockDim.x + 1);

  fillPtr<<<gridDim, blockDim>>>(aFlat, cFlat, pointNum, n * n, a, ainv);
  ASSERT_CUDA_NO_ERROR();
  int *info;
  cudaMalloc(&info, pointNum * sizeof(int));
  Wrapper::cublasGmatinvBatched::call(handle, n, a, n, ainv, n, info, pointNum);
  ASSERT_CUDA_NO_ERROR();
  cudaDeviceSynchronize();

  cudaFree(a);
  cudaFree(ainv);
  cudaFree(info);
}

template <typename T>
void invertDistributed(const std::vector<T *> aFlat, int n, const int pointNum,
                       std::vector<T *> cFlat) {
  const auto &handle = HandleManager::get_cublasHandle();
  const auto worldSize = MemoryPool::getWorldSize();

  std::vector<const T **> a{static_cast<std::size_t>(worldSize)};
  std::vector<T **> ainv{static_cast<std::size_t>(worldSize)};
  std::vector<int *> info{static_cast<std::size_t>(worldSize)};
  dim3 blockDim(std::min(decltype(pointNum)(256), pointNum));
  dim3 gridDim((pointNum - 1) / blockDim.x + 1);
  ASSERT_CUDA_NO_ERROR();

  for (int i = 0; i < worldSize; ++i) {
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&a[i]),
                               pointNum * sizeof(T *), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&ainv[i]),
                               pointNum * sizeof(T *), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&info[i]),
                               pointNum * sizeof(int), i);
  }

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    fillPtr<<<gridDim, blockDim>>>(aFlat[i], cFlat[i], pointNum, n * n, a[i],
                                   ainv[i]);
    ASSERT_CUDA_NO_ERROR();
    Wrapper::cublasGmatinvBatched::call(handle[i], n, a[i], n, ainv[i], n,
                                        info[i], pointNum);
  }
  ASSERT_CUDA_NO_ERROR();
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    MemoryPool::deallocateNormal(info[i], i);
    MemoryPool::deallocateNormal(ainv[i], i);
    MemoryPool::deallocateNormal(a[i], i);
  }
}

template <typename T, int result_weight = 1, int dest_weight = 0>
__global__ void oursGgemvBatched(const T *csrVal, const T *r, int batchSize,
                                 T *dx) {
  /*
blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in
this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize)
    return;

  T *smem = Wrapper::Shared_Memory<T>::get();
  T sum = 0;
  smem[threadIdx.x + threadIdx.y * blockDim.x] =
      r[threadIdx.x + tid * blockDim.x];
  __syncthreads();
  for (unsigned int i = 0; i < blockDim.x; ++i) {
    sum +=
        csrVal[i + threadIdx.x * blockDim.x + tid * blockDim.x * blockDim.x] *
        smem[i + threadIdx.y * blockDim.x];
  }
  dx[threadIdx.x + tid * blockDim.x] =
      result_weight * sum + dest_weight * dx[threadIdx.x + tid * blockDim.x];
}

template <typename T>
bool PreconditionedConjugateGradientSolverLargeSchurDistributedCUDA(
    const std::vector<T *> &SpMVbuffer, std::size_t maxIter,
    double solverRefuseRatio, const double tol, const int cameraNum,
    const int pointNum, const int cameraDim, const int pointDim,
    const std::vector<int> &hplNnz, const int hppRows, const int hllRows,
    const std::vector<T *> &hppCsrVal, const std::vector<T *> &hplCsrVal,
    const std::vector<int *> &hplCsrColInd,
    const std::vector<int *> &hplCsrRowPtr, const std::vector<T *> &hlpCsrVal,
    const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &g,
    const std::vector<T *> &d_x) {
  const auto &comms = HandleManager::get_ncclComm();
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;
  const auto &cusparseHandle = HandleManager::get_cusparseHandle();
  const auto &cublasHandle = HandleManager::get_cublasHandle();
  std::vector<cudaStream_t> cusparseStream, cublasStream;
  const T one{1.0}, zero{0.0}, neg_one{-1.0};
  T alphaN, alphaNegN, rhoNm1;
  std::vector<T> dot;
  std::vector<T *> hppInvCsrVal, pN, rN, axN, temp, deltaXBackup;
  std::vector<cusparseSpMatDescr_t> hpl, hlp;
  std::vector<cusparseDnVecDescr_t> vecx, vecp, vecAx, vectemp;
  cusparseStream.resize(worldSize);
  cublasStream.resize(worldSize);
  dot.resize(worldSize);
  hppInvCsrVal.resize(worldSize);
  pN.resize(worldSize);
  rN.resize(worldSize);
  axN.resize(worldSize);
  temp.resize(worldSize);
  deltaXBackup.resize(worldSize);
  hpl.resize(worldSize);
  hlp.resize(worldSize);
  vecx.resize(worldSize);
  vecp.resize(worldSize);
  vecAx.resize(worldSize);
  vectemp.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
    cublasGetStream_v2(cublasHandle[i], &cublasStream[i]);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&hppInvCsrVal[i]),
                               hppRows * cameraDim * sizeof(T), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&pN[i]),
                               hppRows * sizeof(T), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&rN[i]),
                               hppRows * sizeof(T), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&axN[i]),
                               hppRows * sizeof(T), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&temp[i]),
                               hllRows * sizeof(T), i);

    MemoryPool::allocateNormal(reinterpret_cast<void **>(&deltaXBackup[i]),
                               hllRows * sizeof(T), i);

    cudaMemcpyAsync(rN[i], g[i], hppRows * sizeof(T), cudaMemcpyDeviceToDevice);

    /* Wrap raw data into cuSPARSE generic API objects */
    cusparseCreateCsr(&hpl[i], hppRows, hllRows, hplNnz[i], hplCsrRowPtr[i],
                      hplCsrColInd[i], hplCsrVal[i], CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      cudaDataType);
    cusparseCreateCsr(&hlp[i], hllRows, hppRows, hplNnz[i], hlpCsrRowPtr[i],
                      hlpCsrColInd[i], hlpCsrVal[i], CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      cudaDataType);
    cusparseCreateDnVec(&vecx[i], hppRows, d_x[i], cudaDataType);
    cusparseCreateDnVec(&vecp[i], hppRows, pN[i], cudaDataType);
    cusparseCreateDnVec(&vecAx[i], hppRows, axN[i], cudaDataType);
    cusparseCreateDnVec(&vectemp[i], hllRows, temp[i], cudaDataType);
  }

  invertDistributed(hppCsrVal, cameraDim, cameraNum, hppInvCsrVal);

  /* Allocate workspace for cuSPARSE */
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    /* Begin CG */
    // x1 = ET*x
    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                 hlp[i], vecx[i], &zero, vectemp[i], cudaDataType,
                 CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
  }
  ASSERT_CUDA_NO_ERROR();

  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(temp[i], temp[i], hllRows,
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();

  for (int i = 0; i < worldSize; ++i) {
    dim3 block(pointDim, std::min(32, pointNum));
    dim3 grid((pointNum - 1) / block.y + 1);
    cudaSetDevice(i);
    // borrow pN as temp workspace
    oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T),
                       cusparseStream[i]>>>(hllInvCsrVal[i], temp[i], pointNum,
                                            temp[i]);

    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                 hpl[i], vectemp[i], &zero, vecAx[i], cudaDataType,
                 CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
  }

  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(axN[i], axN[i], hppRows,
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();

  for (int i = 0; i < worldSize; ++i) {
    dim3 block(cameraDim, std::min(32, cameraNum));
    dim3 grid((cameraNum - 1) / block.y + 1);
    cudaSetDevice(i);
    oursGgemvBatched<T, 1, -1>
        <<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(
            hppCsrVal[i], d_x[i], cameraNum, axN[i]);
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(cusparseStream[i]);
    // r = b - Ax
    Wrapper::cublasGaxpy::call(cublasHandle[i], hppRows, &neg_one, axN[i], 1,
                               rN[i], 1);
  }
  int n{0};
  T rhoN{0};
  T rhoMinimum = INFINITY;
  std::vector<T> rho_n_item;
  rho_n_item.resize(worldSize);
  bool done{false};
  do {
    std::size_t offset{0};
    rhoN = 0;
    for (int i = 0; i < worldSize; ++i) {
      dim3 block(cameraDim, std::min(32, cameraNum));
      dim3 grid((cameraNum - 1) / block.y + 1);
      cudaSetDevice(i);
      // borrow axN
      oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T),
                         cublasStream[i]>>>(hppInvCsrVal[i], rN[i], cameraNum,
                                            axN[i]);

      // rhoN = rTr
      const auto nElm = MemoryPool::getElmNum(i, hppRows);
      Wrapper::cublasGdot::call(cublasHandle[i], nElm, &rN[i][offset], 1,
                                &axN[i][offset], 1, &rho_n_item[i]);
      offset += nElm;
    }
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaStreamSynchronize(cublasStream[i]);
      rhoN += rho_n_item[i];
    }
    if (rhoN > solverRefuseRatio * rhoMinimum) {
      for (int i = 0; i < worldSize; ++i) {
        cudaSetDevice(i);
        cudaMemcpyAsync(d_x[i], deltaXBackup[i], hppRows * sizeof(T),
                        cudaMemcpyDeviceToDevice);
      }
      break;
    }
    rhoMinimum = std::min(rhoMinimum, rhoN);

    if (n >= 1) {
      T beta_n = rhoN / rhoNm1;
      for (int i = 0; i < worldSize; ++i) {
        dim3 block(std::min(256, hppRows));
        dim3 grid((hppRows - 1) / block.x + 1);
        cudaSetDevice(i);
        weightedPlusKernel<T>
            <<<grid, block>>>(hppRows, axN[i], pN[i], beta_n, pN[i]);
      }
    } else {
      for (int i = 0; i < worldSize; ++i) {
        cudaSetDevice(i);
        Wrapper::cublasGcopy::call(cublasHandle[i], hppRows, axN[i], 1, pN[i],
                                   1);
      }
    }

    for (int i = 0; i < worldSize; ++i) {
      // Ax = Ad ???? q = Ad
      // x1 = ET*x
      cudaSetDevice(i);
      cudaStreamSynchronize(cublasStream[i]);
      cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                   hlp[i], vecp[i], &zero, vectemp[i], cudaDataType,
                   CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
    }

    ncclGroupStart();
    for (int i = 0; i < worldSize; ++i) {
      ncclAllReduce(temp[i], temp[i], hllRows,
                    Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                    comms[i], cusparseStream[i]);
    }
    ncclGroupEnd();

    for (int i = 0; i < worldSize; ++i) {
      dim3 block(pointDim, std::min(32, pointNum));
      dim3 grid((pointNum - 1) / block.y + 1);
      cudaSetDevice(i);
      // borrow pN as temp workspace
      oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T),
                         cusparseStream[i]>>>(hllInvCsrVal[i], temp[i],
                                              pointNum, temp[i]);

      cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &one,
                   hpl[i], vectemp[i], &zero, vecAx[i], cudaDataType,
                   CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
    }

    ncclGroupStart();
    for (int i = 0; i < worldSize; ++i) {
      ncclAllReduce(axN[i], axN[i], hppRows,
                    Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                    comms[i], cusparseStream[i]);
    }
    ncclGroupEnd();

    for (int i = 0; i < worldSize; ++i) {
      dim3 block(cameraDim, std::min(32, cameraNum));
      dim3 grid((cameraNum - 1) / block.y + 1);
      cudaSetDevice(i);
      oursGgemvBatched<T, 1, -1>
          <<<grid, block, block.x * block.y * sizeof(T), cusparseStream[i]>>>(
              hppCsrVal[i], pN[i], cameraNum, axN[i]);
    }

    offset = 0;
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaStreamSynchronize(cusparseStream[i]);
      // dot :dTq
      const auto nElm = MemoryPool::getElmNum(i, hppRows);
      Wrapper::cublasGdot::call(cublasHandle[i], nElm, &pN[i][offset], 1,
                                &axN[i][offset], 1, &dot[i]);
      offset += nElm;
    }
    // beta_n: one = rhoN / dTq
    double dot_sum{0};
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaStreamSynchronize(cublasStream[i]);
      dot_sum += dot[i];
    }
    alphaN = rhoN / dot_sum;
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      // x=x+alphaN*pN
      cudaMemcpyAsync(deltaXBackup[i], d_x[i], hppRows * sizeof(T),
                      cudaMemcpyDeviceToDevice);
      Wrapper::cublasGaxpy::call(cublasHandle[i], hppRows, &alphaN, pN[i], 1,
                                 d_x[i], 1);
    }

    alphaNegN = -alphaN;

    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      // r = r - alphaN*Ax = r - alphaN*q
      Wrapper::cublasGaxpy::call(cublasHandle[i], hppRows, &alphaNegN, axN[i],
                                 1, rN[i], 1);
    }
    rhoNm1 = rhoN;
    // printf("iteration = %3d, residual = %f\n", n, std::abs(rhoN));
    ++n;
    done = std::abs(rhoN) < tol;
  } while (!done && n < maxIter);
  // cudaSetDevice(0);
  // PRINT_DMEMORY_SEGMENT(d_x[0], 0, 2, T);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cusparseDestroySpMat(hpl[i]);
    cusparseDestroySpMat(hlp[i]);
    cusparseDestroyDnVec(vecx[i]);
    cusparseDestroyDnVec(vecAx[i]);
    cusparseDestroyDnVec(vecp[i]);
    cusparseDestroyDnVec(vectemp[i]);

    MemoryPool::deallocateNormal(deltaXBackup[i], i);
    MemoryPool::deallocateNormal(temp[i], i);
    MemoryPool::deallocateNormal(axN[i], i);
    MemoryPool::deallocateNormal(rN[i], i);
    MemoryPool::deallocateNormal(pN[i], i);
    MemoryPool::deallocateNormal(hppInvCsrVal[i], i);
  }
  return done;
}

template <typename T>
void SchurMakeVDistributed(std::vector<T *> *SpMVbuffer, const int pointNum,
                           const int pointDim, const std::vector<int> &hplNnz,
                           const int hppRows, const int hllRows,
                           const std::vector<T *> &hplCsrVal,
                           const std::vector<int *> &hplCsrColInd,
                           const std::vector<int *> &hplCsrRowPtr,
                           const std::vector<T *> &hllInvCsrVal,
                           const std::vector<T *> &r) {
  const auto &comms = HandleManager::get_ncclComm();
  const auto worldSize = MemoryPool::getWorldSize();
  const auto &cusparseHandle = HandleManager::get_cusparseHandle();
  constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;

  std::vector<T *> v, w;
  std::vector<cudaStream_t> cusparseStream;
  std::vector<cusparseDnVecDescr_t> vecv, vecw;
  std::vector<cusparseSpMatDescr_t> hpl;
  v.resize(worldSize);
  w.resize(worldSize);
  cusparseStream.resize(worldSize);
  vecv.resize(worldSize);
  vecw.resize(worldSize);
  hpl.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
    v[i] = &r[i][0];
    w[i] = &r[i][hppRows];
    cusparseCreateDnVec(&vecv[i], hppRows, v[i], cudaDataType);
    cusparseCreateDnVec(&vecw[i], hllRows, w[i], cudaDataType);
    cusparseCreateCsr(&hpl[i], hppRows, hllRows, hplNnz[i], hplCsrRowPtr[i],
                      hplCsrColInd[i], hplCsrVal[i], CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      cudaDataType);
  }

  dim3 blockDim(pointDim, std::min(32, pointNum));
  dim3 gridDim((pointNum - 1) / blockDim.y + 1);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    // notably, w here is changed(w = C^{-1}w),
    // so later w = C^{-1}(w - ETv) = C^{-1}w - C^{-1}ETv -> w = w - C^{-1}ETv
    oursGgemvBatched<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T),
                       cusparseStream[i]>>>(hllInvCsrVal[i], w[i], pointNum,
                                            w[i]);
  }

  T alpha{-1.0}, beta = T(1. / worldSize);

  SpMVbuffer->resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    // PRINT_DMEMORY(hplCsrVal[i], hplNnz[i], T);
    // PRINT_DMEMORY(hplCsrColInd[i], hplNnz[i], int);
    // PRINT_DMEMORY(hplCsrRowPtr[i], hppRows + 1, int);
    // PRINT_DCSR(hplCsrVal[i], hplCsrColInd[i], hplCsrRowPtr[i], hppRows, T);
    // PRINT_DMEMORY(w[i], hllRows, T);
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, hpl[i], vecw[i], &beta, vecv[i],
                            cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bufferSize);
    MemoryPool::allocateNormal(
        reinterpret_cast<void **>(&SpMVbuffer->operator[](i)),
        bufferSize, i);
    // cudaMalloc(&SpMVbuffer[i], bufferSize);
    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                 hpl[i], vecw[i], &beta, vecv[i], cudaDataType,
                 CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer->operator[](i));
    // PRINT_DMEMORY(v[i], hppRows, T);
  }
  ASSERT_CUDA_NO_ERROR();

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(cusparseStream[i]);

    cusparseDestroySpMat(hpl[i]);
    cusparseDestroyDnVec(vecv[i]);
    cusparseDestroyDnVec(vecw[i]);
  }
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(v[i], v[i], hppRows,
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();
  // cudaSetDevice(0);
  // PRINT_DMEMORY(v[0], hppRows, T);
}

template <typename T>
void SchurSolveWDistributed(
    const std::vector<T *> &SpMVbuffer, const int pointNum, const int pointDim,
    const std::vector<int> &hplNnz, const int hppRows, const int hllRows,
    const std::vector<T *> &hlpCsrVal, const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &d_r,
    const std::vector<T *> &d_x) {
  const auto comms = HandleManager::get_ncclComm();
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declared_cudaDatatype<T>::cuda_dtype;

  std::vector<T *> xc, xp, w;
  xc.resize(worldSize);
  xp.resize(worldSize);
  w.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    xc[i] = &d_x[i][0];
    xp[i] = &d_x[i][hppRows];
    w[i] = &d_r[i][hppRows];
  }

  const auto &cusparseHandle = HandleManager::get_cusparseHandle();

  std::vector<cudaStream_t> cusparseStream;
  std::vector<cusparseDnVecDescr_t> vecxc, vecxp, vecw;
  std::vector<cusparseSpMatDescr_t> hlp;
  cusparseStream.resize(worldSize);
  vecxc.resize(worldSize);
  vecxp.resize(worldSize);
  vecw.resize(worldSize);
  hlp.resize(worldSize);

  for (int i = 0; i < worldSize; ++i) {
    cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);

    cusparseCreateDnVec(&vecxc[i], hppRows, xc[i], cudaDataType);
    cusparseCreateDnVec(&vecxp[i], hllRows, xp[i], cudaDataType);
    cusparseCreateDnVec(&vecw[i], hllRows, w[i], cudaDataType);
    cusparseCreateCsr(&hlp[i], hllRows, hppRows, hplNnz[i], hlpCsrRowPtr[i],
                      hlpCsrColInd[i], hlpCsrVal[i], CUSPARSE_INDEX_32I,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                      cudaDataType);
  }

  T alpha{1.0}, beta{0.0};
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    // x1 = ET*x
    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                 hlp[i], vecxc[i], &beta, vecxp[i], cudaDataType,
                 CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
  }

  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(xp[i], xp[i], hllRows,
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();

  dim3 blockDim(pointDim, std::min(32, pointNum));
  dim3 gridDim((pointNum - 1) / blockDim.y + 1);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    oursGgemvBatched<T, -1, 1>
        <<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T),
           cusparseStream[i]>>>(hllInvCsrVal[i], xp[i], pointNum, w[i]);
    cudaMemcpyAsync(xp[i], w[i], hllRows * sizeof(T), cudaMemcpyDeviceToDevice,
                    cusparseStream[i]);
  }

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(cusparseStream[i]);

    cusparseDestroySpMat(hlp[i]);
    cusparseDestroyDnVec(vecxc[i]);
    cusparseDestroyDnVec(vecw[i]);
  }
}
}  // namespace

template <typename T>
bool SchurSolverDistributed(
    double tol, double solverRefuseRatio, std::size_t maxIter,
    const std::vector<T *> &hppCsrVal, const std::vector<T *> &hllCsrVal,
    const std::vector<T *> &hplCsrVal, const std::vector<int *> &hplCsrColInd,
    const std::vector<int *> &hplCsrRowPtr, const std::vector<T *> &hlpCsrVal,
    const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr, const std::vector<T *> &g,
    int cameraDim, int cameraNum, int pointDim, int pointNum,
    const std::vector<int> &hplNnz, int hppRows, int hllRows,
    const std::vector<T *> &deltaX) {
  ASSERT_CUDA_NO_ERROR();
  // hll inverse-----------------------------------------------------------
  const auto worldSize = MemoryPool::getWorldSize();

  ASSERT_CUDA_NO_ERROR();
  // cudaSetDevice(0);
  // PRINT_DMEMORY(deltaX[0], 9, T);
  std::vector<T *> SpMVbuffer;

  std::vector<T *> hllInvCsrVal;
  hllInvCsrVal.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&hllInvCsrVal[i]),
                               hllRows * pointDim * sizeof(T), i);
  }
  invertDistributed(hllCsrVal, pointDim, pointNum, hllInvCsrVal);

  // cudaSetDevice(0);
  // PRINT_DMEMORY(hllInvCsrVal[0], 9, T);
  ASSERT_CUDA_NO_ERROR();

  SchurMakeVDistributed(&SpMVbuffer, pointNum, pointDim, hplNnz, hppRows,
                        hllRows, hplCsrVal, hplCsrColInd, hplCsrRowPtr,
                        hllInvCsrVal, g);
  bool PCG_success =
      PreconditionedConjugateGradientSolverLargeSchurDistributedCUDA(
          SpMVbuffer, maxIter, solverRefuseRatio, tol, cameraNum, pointNum,
          cameraDim, pointDim, hplNnz, hppRows, hllRows,
          hppCsrVal,
          hplCsrVal, hplCsrColInd, hplCsrRowPtr,
          hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr,
          hllInvCsrVal, g, deltaX);
  SchurSolveWDistributed(SpMVbuffer, pointNum, pointDim, hplNnz, hppRows,
                         hllRows, hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr,
                         hllInvCsrVal, g, deltaX);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    MemoryPool::deallocateNormal(SpMVbuffer[i], i);
    MemoryPool::deallocateNormal(hllInvCsrVal[i], i);
  }
  // cudaSetDevice(0);
  // PRINT_DMEMORY(deltaX[0], 9, T);
  return PCG_success;
}

template <typename T>
bool BaseProblem<T>::solveLinearCUDA(double tol, double solverRefuseRatio,
                                     std::size_t maxIter) {
  bool success;

  if (option.useSchur) {
    // TODO(Jie Ren): need great change
    const auto worldSize = MemoryPool::getWorldSize();
    std::vector<T *> hppCsrVal{static_cast<std::size_t>(worldSize)};
    std::vector<T *> hllCsrVal{static_cast<std::size_t>(worldSize)};
    std::vector<T *> hplCsrVal{static_cast<std::size_t>(worldSize)};
    std::vector<T *> hlpCsrVal{static_cast<std::size_t>(worldSize)};
    std::vector<int *> hplCsrColInd{static_cast<std::size_t>(worldSize)};
    std::vector<int *> hlpCsrColInd{static_cast<std::size_t>(worldSize)};
    std::vector<int *> hplCsrRowPtr{static_cast<std::size_t>(worldSize)};
    std::vector<int *> hlpCsrRowPtr{static_cast<std::size_t>(worldSize)};
    std::vector<T *> g{static_cast<std::size_t>(worldSize)};
    int cameraDim;
    int cameraNum;
    int pointDim;
    int pointNum;
    std::vector<int> hplNnz{};
    hplNnz.resize(worldSize);
    int hppRows;
    int hllRows;
    std::vector<T *> delta_x{static_cast<std::size_t>(worldSize)};

    for (int i = 0; i < worldSize; ++i) {
      auto &schurEquationContainer = edges.schurEquationContainer[i];
      hppCsrVal[i] = schurEquationContainer.csrVal[2];
      hllCsrVal[i] = schurEquationContainer.csrVal[3];
      hplCsrVal[i] = schurEquationContainer.csrVal[0];
      hlpCsrVal[i] = schurEquationContainer.csrVal[1];
      hplCsrColInd[i] = schurEquationContainer.csrColInd[0];
      hlpCsrColInd[i] = schurEquationContainer.csrColInd[1];
      hplCsrRowPtr[i] = schurEquationContainer.csrRowPtr[0];
      hlpCsrRowPtr[i] = schurEquationContainer.csrRowPtr[1];
      g[i] = schurEquationContainer.g;
      cameraDim = schurEquationContainer.dim[0];
      cameraNum = schurEquationContainer.nnz[2] /
                  schurEquationContainer.dim[0] / schurEquationContainer.dim[0];
      pointDim = schurEquationContainer.dim[1];
      pointNum = schurEquationContainer.nnz[3] / schurEquationContainer.dim[1] /
                 schurEquationContainer.dim[1];
      hplNnz[i] = schurEquationContainer.nnz[0];
      hppRows = schurEquationContainer.nnz[2] / schurEquationContainer.dim[0];
      hllRows = schurEquationContainer.nnz[3] / schurEquationContainer.dim[1];
      delta_x[i] = schurDeltaXPtr[i];
    }

    success = SchurSolverDistributed(
        tol, solverRefuseRatio, maxIter, hppCsrVal, hllCsrVal, hplCsrVal,
        hplCsrColInd, hplCsrRowPtr, hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr, g,
        cameraDim, cameraNum, pointDim, pointNum, hplNnz, hppRows, hllRows,
        delta_x);
  } else {
    // TODO(Jie Ren): implement this
  }
  return success;
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
