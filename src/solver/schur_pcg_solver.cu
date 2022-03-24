/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/schur_LM_linear_system.h"
#include "solver/schur_pcg_solver.h"
#include "wrapper.hpp"

#if __CUDACC_VER_MAJOR__ < 11 || \
    (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 2)
#define CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#endif

namespace MegBA {
namespace {
template <typename T>
__global__ void weightedPlusKernel(int nItem, const T *x, const T *y, T weight,
                                   T *z) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem) return;
  z[tid] = x[tid] + weight * y[tid];
}

template <typename T>
__global__ void fillPtr(const T *aData, T *ainvData, const int batchSize,
                        const int hRowsNumPow2, const T **a, T **ainv) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= batchSize) return;
  a[tid] = &aData[tid * hRowsNumPow2];
  ainv[tid] = &ainvData[tid * hRowsNumPow2];
}

template <typename T>
void invert(const T *aFlat, int n, const int num, T *cFlat) {
  cublasHandle_t handle = HandleManager::getCUBLASHandle()[0];

  const T **a;
  T **ainv;
  cudaMalloc(&a, num * sizeof(T *));
  cudaMalloc(&ainv, num * sizeof(T *));
  dim3 blockDim(std::min(decltype(num)(256), num));
  dim3 gridDim((num - 1) / blockDim.x + 1);

  fillPtr<<<gridDim, blockDim>>>(aFlat, cFlat, num, n * n, a, ainv);

  int *info;
  cudaMalloc(&info, num * sizeof(int));
  Wrapper::cublasGmatinvBatched::call(handle, n, a, n, ainv, n, info, num);

  cudaDeviceSynchronize();

  cudaFree(a);
  cudaFree(ainv);
  cudaFree(info);
}

template <typename T>
void invertDistributed(const std::vector<T *> &aFlat, int n, const int num,
                       std::vector<T *> &cFlat) {
  const auto &handle = HandleManager::getCUBLASHandle();
  const auto worldSize = MemoryPool::getWorldSize();

  std::vector<const T **> a{static_cast<std::size_t>(worldSize)};
  std::vector<T **> ainv{static_cast<std::size_t>(worldSize)};
  std::vector<int *> info{static_cast<std::size_t>(worldSize)};
  dim3 blockDim(std::min(decltype(num)(256), num));
  dim3 gridDim((num - 1) / blockDim.x + 1);

  for (int i = 0; i < worldSize; ++i) {
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&a[i]),
                               num * sizeof(T *), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&ainv[i]),
                               num * sizeof(T *), i);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&info[i]),
                               num * sizeof(int), i);
  }

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    fillPtr<<<gridDim, blockDim>>>(aFlat[i], cFlat[i], num, n * n, a[i],
                                   ainv[i]);

    Wrapper::cublasGmatinvBatched::call(handle[i], n, a[i], n, ainv[i], n,
                                        info[i], num);
  }

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
  if (tid >= batchSize) return;

  T *smem = Wrapper::SharedMemory<T>::get();
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
bool schurPCGSolverDistributedCUDA(
    const std::vector<T *> &SpMVbuffer, SolverOption::SolverOptionPCG option,
    const int cameraNum, const int pointNum, const int cameraDim,
    const int pointDim, const std::vector<int> &hplNnz, const int hppRows,
    const int hllRows, const std::vector<T *> &hppCsrVal,
    const std::vector<T *> &hplCsrVal, const std::vector<int *> &hplCsrColInd,
    const std::vector<int *> &hplCsrRowPtr, const std::vector<T *> &hlpCsrVal,
    const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &g,
    const std::vector<T *> &d_x) {
  const auto &comms = HandleManager::getNCCLComm();
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declaredDtype<T>::cudaDtype;
  const auto &cusparseHandle = HandleManager::getCUSPARSEHandle();
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
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

  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(temp[i], temp[i], hllRows,
                  Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                  cusparseStream[i]);
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
    ncclAllReduce(axN[i], axN[i], hppRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
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
      const auto nItem = MemoryPool::getItemNum(i, hppRows);
      Wrapper::cublasGdot::call(cublasHandle[i], nItem, &rN[i][offset], 1,
                                &axN[i][offset], 1, &rho_n_item[i]);
      offset += nItem;
    }
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaStreamSynchronize(cublasStream[i]);
      rhoN += rho_n_item[i];
    }
    if (rhoN > option.refuseRatio * rhoMinimum) {
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
                    Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                    cusparseStream[i]);
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
                    Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                    cusparseStream[i]);
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
      const auto nItem = MemoryPool::getItemNum(i, hppRows);
      Wrapper::cublasGdot::call(cublasHandle[i], nItem, &pN[i][offset], 1,
                                &axN[i][offset], 1, &dot[i]);
      offset += nItem;
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
    done = std::abs(rhoN) < option.tol;
  } while (!done && n < option.maxIter);
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
void schurMakeVDistributed(std::vector<T *> &SpMVbuffer, const int pointNum,
                           const int pointDim, const std::vector<int> &hplNnz,
                           const int hppRows, const int hllRows,
                           const std::vector<T *> &hplCsrVal,
                           const std::vector<int *> &hplCsrColInd,
                           const std::vector<int *> &hplCsrRowPtr,
                           const std::vector<T *> &hllInvCsrVal,
                           const std::vector<T *> &r) {
  const auto &comms = HandleManager::getNCCLComm();
  const auto worldSize = MemoryPool::getWorldSize();
  const auto &cusparseHandle = HandleManager::getCUSPARSEHandle();
  constexpr auto cudaDataType = Wrapper::declaredDtype<T>::cudaDtype;

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

  SpMVbuffer.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    size_t bufferSize = 0;
    cusparseSpMV_bufferSize(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE,
                            &alpha, hpl[i], vecw[i], &beta, vecv[i],
                            cudaDataType, CUSPARSE_SPMV_ALG_DEFAULT,
                            &bufferSize);
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&SpMVbuffer[i]),
                               bufferSize, i);
    cusparseSpMV(cusparseHandle[i], CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha,
                 hpl[i], vecw[i], &beta, vecv[i], cudaDataType,
                 CUSPARSE_SPMV_ALG_DEFAULT, SpMVbuffer[i]);
  }

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(cusparseStream[i]);

    cusparseDestroySpMat(hpl[i]);
    cusparseDestroyDnVec(vecv[i]);
    cusparseDestroyDnVec(vecw[i]);
  }
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(v[i], v[i], hppRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();
}

template <typename T>
void schurSolveWDistributed(
    const std::vector<T *> &SpMVbuffer, const int pointNum, const int pointDim,
    const std::vector<int> &hplNnz, const int hppRows, const int hllRows,
    const std::vector<T *> &hlpCsrVal, const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &d_r,
    const std::vector<T *> &d_x) {
  const auto comms = HandleManager::getNCCLComm();
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declaredDtype<T>::cudaDtype;

  std::vector<T *> xc, xp, w;
  xc.resize(worldSize);
  xp.resize(worldSize);
  w.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    xc[i] = &d_x[i][0];
    xp[i] = &d_x[i][hppRows];
    w[i] = &d_r[i][hppRows];
  }

  const auto &cusparseHandle = HandleManager::getCUSPARSEHandle();

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
    ncclAllReduce(xp[i], xp[i], hllRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
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

template <typename T>
bool SchurPCGSolverDistributed(
    const SolverOption::SolverOptionPCG &option,
    const std::vector<T *> &hppCsrVal, const std::vector<T *> &hllCsrVal,
    const std::vector<T *> &hplCsrVal, const std::vector<int *> &hplCsrColInd,
    const std::vector<int *> &hplCsrRowPtr, const std::vector<T *> &hlpCsrVal,
    const std::vector<int *> &hlpCsrColInd,
    const std::vector<int *> &hlpCsrRowPtr, const std::vector<T *> &g,
    int cameraDim, int cameraNum, int pointDim, int pointNum,
    const std::vector<int> &hplNnz, int hppRows, int hllRows,
    const std::vector<T *> &deltaX) {
  // hll inverse-----------------------------------------------------------
  const auto worldSize = MemoryPool::getWorldSize();

  std::vector<T *> SpMVbuffer;

  std::vector<T *> hllInvCsrVal;
  hllInvCsrVal.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&hllInvCsrVal[i]),
                               hllRows * pointDim * sizeof(T), i);
  }
  invertDistributed(hllCsrVal, pointDim, pointNum, hllInvCsrVal);

  schurMakeVDistributed(SpMVbuffer, pointNum, pointDim, hplNnz, hppRows,
                        hllRows, hplCsrVal, hplCsrColInd, hplCsrRowPtr,
                        hllInvCsrVal, g);
  bool PCG_success = schurPCGSolverDistributedCUDA(
      SpMVbuffer, option, cameraNum, pointNum, cameraDim, pointDim, hplNnz,
      hppRows, hllRows, hppCsrVal, hplCsrVal, hplCsrColInd, hplCsrRowPtr,
      hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr, hllInvCsrVal, g, deltaX);
  schurSolveWDistributed(SpMVbuffer, pointNum, pointDim, hplNnz, hppRows,
                         hllRows, hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr,
                         hllInvCsrVal, g, deltaX);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaDeviceSynchronize();
    MemoryPool::deallocateNormal(SpMVbuffer[i], i);
    MemoryPool::deallocateNormal(hllInvCsrVal[i], i);
  }
  return PCG_success;
}
}  // namespace

template <typename T>
void SchurPCGSolver<T>::solve(const BaseLinearSystem<T> &baseLinearSystem) {
  const auto &linearSystem =
      dynamic_cast<const SchurLinearSystem<T> &>(baseLinearSystem);
  const std::size_t worldSize = linearSystem.problemOption.deviceUsed.size();
  std::vector<T *> hppCsrVal{worldSize};
  std::vector<T *> hllCsrVal{worldSize};
  std::vector<T *> hplCsrVal{worldSize};
  std::vector<T *> hlpCsrVal{worldSize};
  std::vector<int *> hplCsrColInd{worldSize};
  std::vector<int *> hlpCsrColInd{worldSize};
  std::vector<int *> hplCsrRowPtr{worldSize};
  std::vector<int *> hlpCsrRowPtr{worldSize};
  std::vector<T *> g{worldSize};
  std::vector<int> hplNnz{};
  hplNnz.resize(worldSize);
  std::vector<T *> deltaX{worldSize};

  for (int i = 0; i < worldSize; ++i) {
    hppCsrVal[i] = linearSystem.equationContainers[i].csrVal[2];
    hllCsrVal[i] = linearSystem.equationContainers[i].csrVal[3];
    hplCsrVal[i] = linearSystem.equationContainers[i].csrVal[0];
    hlpCsrVal[i] = linearSystem.equationContainers[i].csrVal[1];
    hplCsrColInd[i] = linearSystem.equationContainers[i].csrColInd[0];
    hlpCsrColInd[i] = linearSystem.equationContainers[i].csrColInd[1];
    hplCsrRowPtr[i] = linearSystem.equationContainers[i].csrRowPtr[0];
    hlpCsrRowPtr[i] = linearSystem.equationContainers[i].csrRowPtr[1];
    g[i] = linearSystem.g[i];
    hplNnz[i] = linearSystem.equationContainers[i].nnz[0];
    deltaX[i] = linearSystem.deltaXPtr[i];
  }

  SchurPCGSolverDistributed(
      this->solverOption.solverOptionPCG, hppCsrVal, hllCsrVal, hplCsrVal,
      hplCsrColInd, hplCsrRowPtr, hlpCsrVal, hlpCsrColInd, hlpCsrRowPtr, g,
      linearSystem.dim[0], linearSystem.num[0], linearSystem.dim[1],
      linearSystem.num[1], hplNnz, linearSystem.dim[0] * linearSystem.num[0],
      linearSystem.dim[1] * linearSystem.num[1], deltaX);
}

SPECIALIZE_CLASS(SchurPCGSolver);
}  // namespace MegBA
