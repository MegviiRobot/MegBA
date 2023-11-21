/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/implicit_schur_LM_linear_system.h"
#include "macro.h"
#include "solver/implicit_schur_pcg_solver.h"
#include "wrapper.hpp"

#if __CUDACC_VER_MAJOR__ < 11 || \
    (__CUDACC_VER_MAJOR__ == 11 && __CUDACC_VER_MINOR__ <= 2)
#define CUSPARSE_SPMV_ALG_DEFAULT CUSPARSE_MV_ALG_DEFAULT
#endif

namespace MegBA {
namespace {
template <typename T>
__global__ void implicitEMulx(const T *const *const valPtrs,
                              const T *const xPtrs,
                              const int *absolutePositionCamera,
                              const int *absolutePositionPoint,
                              const int resDim, const int cameraDim,
                              const int pointDim, const int errorNum,
                              T *result) {
  /*
   * make sure that blockDim.x % 32 == 0, if so, there won't be any thread
   * divergence within a wrap.
   */
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= errorNum) return;

  const int absolutePositionPointLocal = absolutePositionPoint[tid];
  const int absolutePositionCameraLocal = absolutePositionCamera[tid];

  for (int i = 0; i < resDim; ++i) {
    T Sum{0.};
    T JpX{0.};
    // Jp * x
    for (int j = cameraDim; j < cameraDim + pointDim; ++j) {
      const T valI = valPtrs[i][errorNum * j + tid];
      JpX +=
          valI * xPtrs[absolutePositionPointLocal * pointDim + j - cameraDim];
    }
    // Jc.T * Jp * x
    for (int j = 0; j < cameraDim; ++j) {
      const T valI = valPtrs[i][errorNum * j + tid];
      Sum = valI * JpX;
      atomicAdd(&result[absolutePositionCameraLocal * cameraDim + j], Sum);
    }
  }
}

template <typename T>
__global__ void implicitETMulx(const T *const *const valPtrs,
                               const T *const xPtrs,
                               const int *absolutePositionCamera,
                               const int *absolutePositionPoint,
                               const int resDim, const int cameraDim,
                               const int pointDim, const int errorNum,
                               T *result) {
  /*
   * make sure that blockDim.x % 32 == 0, if so, there won't be any thread
   * divergence within a wrap.
   */
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= errorNum) return;

  const int absolutePositionPointLocal = absolutePositionPoint[tid];
  const int absolutePositionCameraLocal = absolutePositionCamera[tid];

  for (int i = 0; i < resDim; ++i) {
    T Sum{0.};
    T JcX{0.};
    // Jc * x
    for (int j = 0; j < cameraDim; ++j) {
      const T valI = valPtrs[i][errorNum * j + tid];
      JcX += valI * xPtrs[absolutePositionCameraLocal * cameraDim + j];
    }
    // Jp.T * Jc * x
    for (int j = cameraDim; j < cameraDim + pointDim; ++j) {
      const T valI = valPtrs[i][errorNum * j + tid];
      Sum = valI * JcX;
      atomicAdd(&result[absolutePositionPointLocal * pointDim + j - cameraDim],
                Sum);
    }
  }
}

template <typename T>
__global__ void weightedPlusKernel(int nItem, const T *x, const T *y, T weight,
                                   T *z) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem) return;
  z[tid] = x[tid] + weight * y[tid];
}

template <typename T>
__global__ void doubleWeightedPlusKernel(int nItem, T weightx, const T *x,
                                         T weighty, const T *y, T *z) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem) return;
  z[tid] = weightx * x[tid] + weighty * y[tid];
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
bool ImplicitSchurPCGSolverDistributedCUDA(
    SolverOption::SolverOptionPCG option,
    const std::vector<const T **> &valPtrsDevice, const EdgeVector<T> &edges,
    const int resDim, const int cameraNum, const int pointNum,
    const int cameraDim, const int pointDim, const int hppRows,
    const int hllRows, const std::vector<T *> &hppCsrVal,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &g,
    const std::vector<T *> &d_x) {
#ifdef MEGBA_ENABLE_NCCL
  const auto &comms = HandleManager::getNCCLComm();
#endif
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declaredDtype<T>::cudaDtype;
  const auto &cusparseHandle = HandleManager::getCUSPARSEHandle();
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  std::vector<cudaStream_t> cusparseStream, cublasStream;
  const T one{1.0}, zero{0.0}, neg_one{-1.0};
  T alphaN, alphaNegN, rhoNm1;
  std::vector<T> dot;
  std::vector<T *> hppInvCsrVal, pN, rN, axN, temp, deltaXBackup;
  cusparseStream.resize(worldSize);
  cublasStream.resize(worldSize);
  dot.resize(worldSize);
  hppInvCsrVal.resize(worldSize);
  pN.resize(worldSize);
  rN.resize(worldSize);
  axN.resize(worldSize);
  temp.resize(worldSize);
  deltaXBackup.resize(worldSize);
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
  }

  invertDistributed(hppCsrVal, cameraDim, cameraNum, hppInvCsrVal);

  /* Allocate workspace for cuSPARSE */
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    /* Begin CG */
    // x1 = ET*x
    const auto &positionContainer = edges.getPositionContainers()[i];
    const auto edgeNum = MemoryPool::getItemNum(i);
    dim3 block(std::min((decltype(edgeNum))32, edgeNum));
    dim3 grid((edgeNum - 1) / block.x + 1);
    cudaMemsetAsync(temp[i], 0, hllRows * sizeof(T), cusparseStream[i]);
    implicitETMulx<<<grid, block, 0, cusparseStream[i]>>>(
        valPtrsDevice[i], d_x[i], positionContainer.absolutePosition[0],
        positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
        edgeNum, temp[i]);
  }

#ifdef MEGBA_ENABLE_NCCL
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(temp[i], temp[i], hllRows,
                  Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                  cusparseStream[i]);
  }
  ncclGroupEnd();
#endif

  for (int i = 0; i < worldSize; ++i) {
    dim3 block(pointDim, std::min(32, pointNum));
    dim3 grid((pointNum - 1) / block.y + 1);
    cudaSetDevice(i);
    oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T),
                       cusparseStream[i]>>>(hllInvCsrVal[i], temp[i], pointNum,
                                            temp[i]);

    const auto &positionContainer = edges.getPositionContainers()[i];
    const auto edgeNum = MemoryPool::getItemNum(i);
    dim3 block_(std::min((decltype(edgeNum))32, edgeNum));
    dim3 grid_((edgeNum - 1) / block_.x + 1);
    cudaMemsetAsync(axN[i], 0, hppRows * sizeof(T), cusparseStream[i]);
    implicitEMulx<<<grid_, block_, 0, cusparseStream[i]>>>(
        valPtrsDevice[i], temp[i], positionContainer.absolutePosition[0],
        positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
        edgeNum, axN[i]);
  }

#ifdef MEGBA_ENABLE_NCCL
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(axN[i], axN[i], hppRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();
#endif

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
      const auto &positionContainer = edges.getPositionContainers()[i];
      const auto edgeNum = MemoryPool::getItemNum(i);
      dim3 block(std::min((decltype(edgeNum))32, edgeNum));
      dim3 grid((edgeNum - 1) / block.x + 1);
      cudaMemsetAsync(temp[i], 0, hllRows * sizeof(T), cusparseStream[i]);
      implicitETMulx<<<grid, block, 0, cusparseStream[i]>>>(
          valPtrsDevice[i], pN[i], positionContainer.absolutePosition[0],
          positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
          edgeNum, temp[i]);
    }

#ifdef MEGBA_ENABLE_NCCL
    ncclGroupStart();
    for (int i = 0; i < worldSize; ++i) {
      ncclAllReduce(temp[i], temp[i], hllRows,
                    Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                    cusparseStream[i]);
    }
    ncclGroupEnd();
#endif

    for (int i = 0; i < worldSize; ++i) {
      dim3 block(pointDim, std::min(32, pointNum));
      dim3 grid((pointNum - 1) / block.y + 1);
      cudaSetDevice(i);
      // borrow pN as temp workspace
      oursGgemvBatched<<<grid, block, block.x * block.y * sizeof(T),
                         cusparseStream[i]>>>(hllInvCsrVal[i], temp[i],
                                              pointNum, temp[i]);

      const auto &positionContainer = edges.getPositionContainers()[i];
      const auto edgeNum = MemoryPool::getItemNum(i);
      dim3 block_(std::min((decltype(edgeNum))32, edgeNum));
      dim3 grid_((edgeNum - 1) / block_.x + 1);
      cudaMemsetAsync(axN[i], 0, hppRows * sizeof(T), cusparseStream[i]);
      implicitEMulx<<<grid_, block_, 0, cusparseStream[i]>>>(
          valPtrsDevice[i], temp[i], positionContainer.absolutePosition[0],
          positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
          edgeNum, axN[i]);
    }

#ifdef MEGBA_ENABLE_NCCL
    ncclGroupStart();
    for (int i = 0; i < worldSize; ++i) {
      ncclAllReduce(axN[i], axN[i], hppRows,
                    Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                    cusparseStream[i]);
    }
    ncclGroupEnd();
#endif

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
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
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
void implicitSchurMakeVDistributed(const std::vector<const T **> &valPtrsDevice,
                                   const int pointNum, const int pointDim,
                                   const int resDim, const int cameraDim,
                                   const int hppRows,
                                   const EdgeVector<T> &edges,
                                   const std::vector<T *> &hllInvCsrVal,
                                   const std::vector<T *> &r) {
#ifdef MEGBA_ENABLE_NCCL
  const auto &comms = HandleManager::getNCCLComm();
#endif
  const auto worldSize = MemoryPool::getWorldSize();
  constexpr auto cudaDataType = Wrapper::declaredDtype<T>::cudaDtype;
  const auto &cusparseHandle = HandleManager::getCUSPARSEHandle();
  std::vector<T *> v, w;
  std::vector<T *> EMulxVal;
  std::vector<cudaStream_t> cusparseStream;
  v.resize(worldSize);
  w.resize(worldSize);
  EMulxVal.resize(worldSize);
  cusparseStream.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
    v[i] = &r[i][0];
    w[i] = &r[i][hppRows];
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&EMulxVal[i]),
                               hppRows * sizeof(T), i);
  }

  dim3 blockDim(pointDim, std::min(32, pointNum));
  dim3 gridDim((pointNum - 1) / blockDim.y + 1);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    // notably, w here is changed(w = C^{-1}w),
    // so later w = C^{-1}(w - ETv) = C^{-1}w - C^{-1}ETv -> w = w - C^{-1}ETv
    // w = C-1 * gpoint
    oursGgemvBatched<<<gridDim, blockDim, blockDim.x * blockDim.y * sizeof(T),
                       cusparseStream[i]>>>(hllInvCsrVal[i], w[i], pointNum,
                                            w[i]);
  }

  T alpha{-1.0}, beta = T(1. / worldSize);

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    const auto &positionContainer = edges.getPositionContainers()[i];
    const auto edgeNum = MemoryPool::getItemNum(i);
    dim3 block_(std::min((decltype(edgeNum))32, edgeNum));
    dim3 grid_((edgeNum - 1) / block_.x + 1);
    // E * C-1 * gpoint = Jc.T * Jp * w, w = C-1 * gpoint
    implicitEMulx<<<grid_, block_, 0, cusparseStream[i]>>>(
        valPtrsDevice[i], w[i], positionContainer.absolutePosition[0],
        positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
        edgeNum, EMulxVal[i]);
    // v = v - Jc.T * Jp * C-1 * gpoint
    dim3 block(std::min(256, hppRows));
    dim3 grid((hppRows - 1) / block.x + 1);
    doubleWeightedPlusKernel<T><<<grid, block, 0, cusparseStream[i]>>>(
        hppRows, alpha, EMulxVal[i], beta, v[i], v[i]);
  }

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(cusparseStream[i]);
    MemoryPool::deallocateNormal(EMulxVal[i], i);
  }
#ifdef MEGBA_ENABLE_NCCL
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(v[i], v[i], hppRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();
#endif
}

template <typename T>
void implicitSchurSolveWDistributed(
    const std::vector<const T **> &valPtrsDevice, const EdgeVector<T> &edges,
    const int resDim, const int pointNum, const int pointDim,
    const int cameraDim, const int hppRows, const int hllRows,
    const std::vector<T *> &hllInvCsrVal, const std::vector<T *> &d_r,
    const std::vector<T *> &d_x) {
#ifdef MEGBA_ENABLE_NCCL
  const auto comms = HandleManager::getNCCLComm();
#endif
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
  cusparseStream.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cusparseGetStream(cusparseHandle[i], &cusparseStream[i]);
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    const auto &positionContainer = edges.getPositionContainers()[i];
    // xp = Jp.T * Jc * xc = E.T * xc
    const auto edgeNum = MemoryPool::getItemNum(i);
    dim3 block(std::min((decltype(edgeNum))32, edgeNum));
    dim3 grid((edgeNum - 1) / block.x + 1);
    cudaMemsetAsync(xp[i], 0, hllRows * sizeof(T), cusparseStream[i]);
    implicitETMulx<<<grid, block, 0, cusparseStream[i]>>>(
        valPtrsDevice[i], xc[i], positionContainer.absolutePosition[0],
        positionContainer.absolutePosition[1], resDim, cameraDim, pointDim,
        edgeNum, xp[i]);
  }

#ifdef MEGBA_ENABLE_NCCL
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(xp[i], xp[i], hllRows, Wrapper::declaredDtype<T>::ncclDtype,
                  ncclSum, comms[i], cusparseStream[i]);
  }
  ncclGroupEnd();
#endif
  // w = w - {C-1} * xp, w = {C-1} * gpoint, xp = ET * deltaXcamera
  // xp = w
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
  }
}

template <typename T>
bool ImplicitSchurPCGSolverDistributed(
    const SolverOption::SolverOptionPCG &option,
    const std::vector<T *> &hppCsrVal, const std::vector<T *> &hllCsrVal,
    const std::vector<const T **> &valPtrsDevice, const std::vector<T *> &g,
    const EdgeVector<T> &edges, int resDim, int cameraDim, int cameraNum,
    int pointDim, int pointNum, int hppRows, int hllRows,
    const std::vector<T *> &deltaX) {
  // hll inverse-----------------------------------------------------------
  const auto worldSize = MemoryPool::getWorldSize();

  std::vector<T *> hllInvCsrVal;
  hllInvCsrVal.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    MemoryPool::allocateNormal(reinterpret_cast<void **>(&hllInvCsrVal[i]),
                               hllRows * pointDim * sizeof(T), i);
  }
  invertDistributed(hllCsrVal, pointDim, pointNum, hllInvCsrVal);

  // v - E * {C-1} * gpoint
  implicitSchurMakeVDistributed(valPtrsDevice, pointNum, pointDim, resDim,
                                cameraDim, hppRows, edges, hllInvCsrVal, g);
  // xc
  bool PCG_success = ImplicitSchurPCGSolverDistributedCUDA(
      option, valPtrsDevice, edges, resDim, cameraNum, pointNum, cameraDim,
      pointDim, hppRows, hllRows, hppCsrVal, hllInvCsrVal, g, deltaX);

  // xp = {C-1} * gpoint - {C-1} * Jp.T * Jc * deltaXcamera
  implicitSchurSolveWDistributed(valPtrsDevice, edges, resDim, pointNum,
                                 pointDim, cameraDim, hppRows, hllRows,
                                 hllInvCsrVal, g, deltaX);

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    MemoryPool::deallocateNormal(hllInvCsrVal[i], i);
  }
  return PCG_success;
}
}  // namespace

template <typename T>
void ImplicitSchurPCGSolver<T>::solve(
    const BaseLinearSystem<T> &baseLinearSystem, const EdgeVector<T> &edges,
    const JVD<T> &jetEstimation) {
  const auto &linearSystem =
      dynamic_cast<const ImplicitSchurLinearSystem<T> &>(baseLinearSystem);
  const std::size_t worldSize = linearSystem.problemOption.deviceUsed.size();
  std::vector<T *> hppCsrVal{worldSize};
  std::vector<T *> hllCsrVal{worldSize};
  std::vector<T *> g{worldSize};
  std::vector<T *> deltaX{worldSize};

  const auto rows = jetEstimation.rows(), cols = jetEstimation.cols();
  const auto resDim = rows * cols;
  std::vector<std::unique_ptr<const T *[]>> totalPtrs {};
  totalPtrs.reserve(worldSize);
  std::vector<const T **> totalPtrsDevice{worldSize};
  std::vector<const T **> valPtrs{worldSize};
  std::vector<const T **> valPtrsDevice{worldSize};

  for (int deviceRank = 0; deviceRank < worldSize; ++deviceRank) {
    hppCsrVal[deviceRank] =
        linearSystem.implicitEquationContainers[deviceRank].csrVal[0];
    hllCsrVal[deviceRank] =
        linearSystem.implicitEquationContainers[deviceRank].csrVal[1];
    g[deviceRank] = linearSystem.g[deviceRank];
    deltaX[deviceRank] = linearSystem.deltaXPtr[deviceRank];

    totalPtrs.emplace_back(new const T *[resDim]);
    cudaSetDevice(deviceRank);
    cudaMalloc(&totalPtrsDevice[deviceRank], resDim * sizeof(T *));

    valPtrs[deviceRank] = &totalPtrs[deviceRank][0];
    valPtrsDevice[deviceRank] = &totalPtrsDevice[deviceRank][0];
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) {
        const auto &jetEstimationInner = jetEstimation(i, j);
        valPtrs[deviceRank][j + i * cols] =
            jetEstimationInner.getCUDAGradPtr()[deviceRank];
      }
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaMemcpyAsync(totalPtrsDevice[i], totalPtrs[i].get(),
                    resDim * sizeof(T *), cudaMemcpyHostToDevice);
    ASSERT_CUDA_NO_ERROR();
  }

  ImplicitSchurPCGSolverDistributed(
      this->solverOption.solverOptionPCG, hppCsrVal, hllCsrVal, valPtrsDevice,
      g, edges, resDim, linearSystem.dim[0], linearSystem.num[0],
      linearSystem.dim[1], linearSystem.num[1],
      linearSystem.dim[0] * linearSystem.num[0],
      linearSystem.dim[1] * linearSystem.num[1], deltaX);
  ASSERT_CUDA_NO_ERROR();

  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(nullptr);
    cudaFree(totalPtrsDevice[i]);
  }
}

SPECIALIZE_STRUCT(ImplicitSchurPCGSolver);
}  // namespace MegBA
