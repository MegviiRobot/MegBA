/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system_manager/schurLM_linear_system_manager.h"
#include <thrust/device_ptr.h>
#include <thrust/async/reduce.h>
#include "wrapper.hpp"

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
union AtomicUnion{
  double dValue;
  unsigned long long ullValue;
};

__inline__ __device__ double atomicAdd(double* address, double val) {
  AtomicUnion old, assumed;
  old.dValue = *address;

  do {
    assumed = old;
    old.ullValue = atomicCAS(reinterpret_cast<unsigned long long *>(address),
                             assumed.ullValue,
                             AtomicUnion{val + assumed.dValue}.ullValue);

    // Note: uses integer comparison to
    // avoid hang in case of NaN (since NaN != NaN)
  } while (assumed.ullValue != old.ullValue);

  return old.dValue;
}
#endif

namespace MegBA {
namespace {
template <typename T>
__device__ void makeHpp(const T *valSmem, const T valI, const int cameraDim,
                        const int hppCsrRowI, T *hppCsrVal) {
  for (int i = 0; i < cameraDim; ++i)
    atomicAdd(&hppCsrVal[hppCsrRowI + i],
              valI * valSmem[i * blockDim.x + threadIdx.x]);
}

template <typename T>
__device__ void makeHpl(const T *valSmem, const T valI,
                        const int relativePositionPoint, const int pointDim,
                        const int cameraDim, int hplCsrRow, T *hplCsrVal) {
  const int hplCsrRowI = hplCsrRow + relativePositionPoint * pointDim;

  for (int i = 0; i < pointDim; ++i) {
    hplCsrVal[hplCsrRowI + i] +=
        valI * valSmem[(i + cameraDim) * blockDim.x + threadIdx.x];
  }
}

template <typename T>
__device__ void makeHlp(const T *valSmem, const T valI,
                        const int relativePositionCamera, const int cameraDim,
                        int hlpCsrRow, T *hlpCsrVal) {
  const int hlpCsrRow_i = hlpCsrRow + relativePositionCamera * cameraDim;

  for (int i = 0; i < cameraDim; ++i) {
    hlpCsrVal[hlpCsrRow_i + i] += valI * valSmem[i * blockDim.x + threadIdx.x];
  }
}

template <typename T>
__device__ void makeHll(const T *valSmem, const T valI, const int pointDim,
                        const int cameraDim, const int hllPosition,
                        T *hllMatrix) {
  for (int i = 0; i < pointDim; ++i)
    atomicAdd(&hllMatrix[hllPosition + i],
              valI * valSmem[(i + cameraDim) * blockDim.x + threadIdx.x]);
}

template <typename T>
__global__ void makeHSchur(
    const T *const *const valPtrs, const T *const *const errorPtrs,
    const int *absolutePositionCamera, const int *absolutePositionPoint,
    const int *relativePositionCamera, const int *relativePositionPoint,
    const int *hplCsrRowPtr, const int *hlpCsrRowPtr, const int resDim,
    const int cameraDim, const int pointDim, const int errorNum, T *gCamera,
    T *gPoint, T *hppCsrVal, T *hllCsrVal, T *hplCsrVal, T *hlpCsrVal) {
  /*
                 * make sure that blockDim.x % 32 == 0, if so, there won't be any thread divergence within a wrap.
   */
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= errorNum) return;

  T *valSmem = Wrapper::Shared_Memory<T>::get();

  const int absolutePositionPointLocal = absolutePositionPoint[tid];
  const int absolutePositionCameraLocal = absolutePositionCamera[tid];
  const int relativePositionPointLocal = relativePositionPoint[tid];
  const int relativePositionCameraLocal = relativePositionCamera[tid];

  T gSum{0.};
  for (int i = 0; i < resDim; ++i) {
    const T valI = valPtrs[i][errorNum * threadIdx.y + tid];
    __syncthreads();
    valSmem[threadIdx.y * blockDim.x + threadIdx.x] = valI;
    __syncthreads();

    if (threadIdx.y < cameraDim) {
      makeHpp(
          valSmem, valI, cameraDim,
          (absolutePositionCameraLocal * cameraDim + threadIdx.y) * cameraDim,
          hppCsrVal);
      makeHpl(
          valSmem, valI, relativePositionPointLocal, pointDim, cameraDim,
          hplCsrRowPtr[absolutePositionCameraLocal * cameraDim + threadIdx.y],
          hplCsrVal);
    } else {
      makeHll(valSmem, valI, pointDim, cameraDim,
              absolutePositionPointLocal * (pointDim * pointDim) +
                  (threadIdx.y - cameraDim) * pointDim /* hllPosition */,
              hllCsrVal);
      makeHlp(valSmem, valI, relativePositionCameraLocal, cameraDim,
              hlpCsrRowPtr[absolutePositionPointLocal * pointDim + threadIdx.y -
                           cameraDim],
              hlpCsrVal);
    }
    gSum += -valI * errorPtrs[i][tid];
  }

  if (threadIdx.y < cameraDim) {
    atomicAdd(&gCamera[absolutePositionCameraLocal * cameraDim + threadIdx.y],
              gSum);
  } else {
    atomicAdd(&gPoint[absolutePositionPointLocal * pointDim + threadIdx.y -
                      cameraDim],
              gSum);
  }
}
}  // namespace

template <typename T>
void SchurLMLinearSystemManager<T>::buildLinearSystemCUDA(
    const JVD<T> &jetEstimation, const JVD<T> &jetInformation) {
  const auto rows = jetEstimation.rows(), cols = jetEstimation.cols();
  const auto cameraDim = dim[0];
  const auto pointDim = dim[1];
  const auto cameraNum = num[0];
  const auto pointNum = num[1];
  const auto hppRows = cameraDim * cameraNum;
  const auto hllRows = pointDim * pointNum;
  const std::size_t worldSize = MemoryPool::getWorldSize();

  std::vector<T *> gCameraDevice{worldSize};
  std::vector<T *> gPointDevice{worldSize};
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaMemsetAsync(equationContainers[i].g, 0,
                    (hppRows + hllRows) * sizeof(T));
    gCameraDevice[i] = &equationContainers[i].g[0];
    gPointDevice[i] = &equationContainers[i].g[hppRows];
    cudaMemsetAsync(equationContainers[i].csrVal[0], 0,
                    equationContainers[i].nnz[0] * sizeof(T));
    cudaMemsetAsync(equationContainers[i].csrVal[1], 0,
                    equationContainers[i].nnz[1] * sizeof(T));
    cudaMemsetAsync(equationContainers[i].csrVal[2], 0,
                    equationContainers[i].nnz[2] * sizeof(T));
    cudaMemsetAsync(equationContainers[i].csrVal[3], 0,
                    equationContainers[i].nnz[3] * sizeof(T));
  }

  const auto resDim = rows * cols;
  std::vector<std::unique_ptr<const T *[]>> totalPtrs {};
  totalPtrs.reserve(worldSize);
  std::vector<const T **> totalPtrsDevice{worldSize};

  std::vector<const T **> valPtrs{worldSize};
  std::vector<const T **> valPtrsDevice{worldSize};

  std::vector<const T **> errorPtrs{worldSize};
  std::vector<const T **> errorPtrsDevice{worldSize};
  for (int deviceRank = 0; deviceRank < worldSize; ++deviceRank) {
    totalPtrs.emplace_back(new const T *[resDim * (3 + resDim)]);
    cudaSetDevice(deviceRank);
    cudaMalloc(&totalPtrsDevice[deviceRank],
               resDim * (3 + resDim) * sizeof(T *));

    valPtrs[deviceRank] = &totalPtrs[deviceRank][0];
    valPtrsDevice[deviceRank] = &totalPtrsDevice[deviceRank][0];

    errorPtrs[deviceRank] = &totalPtrs[deviceRank][resDim];
    errorPtrsDevice[deviceRank] = &totalPtrsDevice[deviceRank][resDim];
    for (int i = 0; i < rows; ++i)
      for (int j = 0; j < cols; ++j) {
        const auto &jetEstimationInner = jetEstimation(i, j);
        valPtrs[deviceRank][j + i * cols] =
            jetEstimationInner.getCUDAGradPtr()[deviceRank];
        errorPtrs[deviceRank][j + i * cols] =
            jetEstimationInner.getCUDAResPtr()[deviceRank];
      }
    cudaMemcpyAsync(totalPtrsDevice[deviceRank], totalPtrs[deviceRank].get(),
                    resDim * 2 * sizeof(T *), cudaMemcpyHostToDevice);
  }

  if (jetInformation.rows() != 0 && jetInformation.cols() != 0) {
    // TODO(Jie Ren): implement this
  } else {
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      const auto edgeNum = MemoryPool::getItemNum(i);
      dim3 block(std::min((decltype(edgeNum))32, edgeNum),
                 cameraDim + pointDim);
      dim3 grid((edgeNum - 1) / block.x + 1);
      makeHSchur<<<grid, block, block.x * block.y * sizeof(T)>>>(
          valPtrsDevice[i], errorPtrsDevice[i],
          positionAndRelationContainers[i].absolutePositionCamera,
          positionAndRelationContainers[i].absolutePositionPoint,
          positionAndRelationContainers[i].relativePositionCamera,
          positionAndRelationContainers[i].relativePositionPoint,
          equationContainers[i].csrRowPtr[0],
          equationContainers[i].csrRowPtr[1], resDim, cameraDim, pointDim,
          edgeNum, gCameraDevice[i], gPointDevice[i],
          equationContainers[i].csrVal[2], equationContainers[i].csrVal[3],
          equationContainers[i].csrVal[0], equationContainers[i].csrVal[1]);
    }
  }
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(nullptr);
    cudaFree(totalPtrsDevice[i]);
  }

  const auto &comms = HandleManager::getNCCLComm();
  ncclGroupStart();
  for (int i = 0; i < worldSize; ++i) {
    ncclAllReduce(equationContainers[i].csrVal[2],
                  equationContainers[i].csrVal[2], equationContainers[i].nnz[2],
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], nullptr);
    ncclAllReduce(equationContainers[i].csrVal[3],
                  equationContainers[i].csrVal[3], equationContainers[i].nnz[3],
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], nullptr);
    ncclAllReduce(gCameraDevice[i], gCameraDevice[i], hppRows + hllRows,
                  Wrapper::declared_cudaDatatype<T>::nccl_dtype, ncclSum,
                  comms[i], nullptr);
  }
  ncclGroupEnd();
}

namespace {
template <typename T>
__global__ void RecoverDiagKernel(const T *in, const T a, const int batchSize,
                                  T *out) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
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
   * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
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
}

template <typename T>
void SchurLMLinearSystemManager<T>::preSolve(const AlgoStatus &algoStatus) {
  if (algoStatus.lmAlgoStatus.recoverDiag) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = equationContainers[i];
      RecoverDiag(extractedDiag[i][0], T(1. / algoStatus.lmAlgoStatus.region),
                  num[0], dim[0], container.csrVal[2]);
      RecoverDiag(extractedDiag[i][1], T(1. / algoStatus.lmAlgoStatus.region),
                  num[1], dim[1], container.csrVal[3]);
    }
  } else {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      auto &container = equationContainers[i];
      extractOldAndApplyNewDiag(T(1. / algoStatus.lmAlgoStatus.region), num[0],
                                dim[0], container.csrVal[2],
                                extractedDiag[i][0]);
      extractOldAndApplyNewDiag(T(1. / algoStatus.lmAlgoStatus.region), num[1],
                                dim[1], container.csrVal[3],
                                extractedDiag[i][1]);
    }
  }
}
template <typename T>
void SchurLMLinearSystemManager<T>::backup() {
  const int hessianShape = getHessianShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaMemcpyAsync(deltaXPtrBackup[i], this->deltaXPtr[i],
                    hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
void SchurLMLinearSystemManager<T>::rollback() {
  const int hessianShape = getHessianShape();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaMemcpyAsync(this->deltaXPtr[i], deltaXPtrBackup[i],
                    hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
  }
}

template <typename T>
void SchurLMLinearSystemManager<T>::applyUpdate(T *xPtr) const {
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  const int hessianShape = getHessianShape();
  const T one = 1.;
  cudaSetDevice(0);
  Wrapper::cublasGaxpy::call(cublasHandle[0], hessianShape, &one,
                             this->deltaXPtr[0], 1, xPtr, 1);
}

namespace {
template <typename T>
__global__ void JdxpF(const T *grad, const T *deltaX, const T *res,
                      const int *absCameraPosition, const int *absPointPosition,
                      const int nItem, const int cameraDim, const int cameraNum,
                      const int pointDim, T *out) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem)
    return;
  T sum{0};
  const int absCameraPositionLocal = absCameraPosition[tid];
  const int absPointPositionLocal = absPointPosition[tid];
  for (int i = 0; i < cameraDim; ++i) {
    sum +=
        grad[tid + i * nItem] * deltaX[i + absCameraPositionLocal * cameraDim];
  }
  for (int i = 0; i < pointDim; ++i) {
    sum += grad[tid + (i + cameraDim) * nItem] *
           deltaX[i + cameraDim * cameraNum + absPointPositionLocal * pointDim];
  }
  out[tid] = (sum + res[tid]) * (sum + res[tid]);
}
}

template <typename T>
double SchurLMLinearSystemManager<T>::computeRhoDenominator(
    JVD<T> &JV, std::vector<T *> &schurDeltaXPtr) {
  T rhoDenominator{0};
  std::vector<std::vector<T *>> Jdx;
  Jdx.resize(MemoryPool::getWorldSize());
  const int cameraDim = dim[0];
  const int cameraNum = num[0];
  const int pointDim = dim[1];

  std::vector<std::vector<thrust::system::cuda::unique_eager_future<T>>>
      futures;
  futures.resize(MemoryPool::getWorldSize());

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = MemoryPool::getItemNum(i);
    const auto &positionContainer = positionAndRelationContainers[i];
    futures[i].resize(JV.size());
    for (int j = 0; j < JV.size(); ++j) {
      auto &J = JV(j);
      T *ptr;
      MemoryPool::allocateNormal(reinterpret_cast<void **>(&ptr),
                                 nItem * sizeof(T), i);
      dim3 block(std::min((std::size_t)256, nItem));
      dim3 grid((nItem - 1) / block.x + 1);
      JdxpF<<<grid, block>>>(J.getCUDAGradPtr()[i], schurDeltaXPtr[i],
                             J.getCUDAResPtr()[i],
                             positionContainer.absolutePositionCamera,
                             positionContainer.absolutePositionPoint,
                             nItem, cameraDim, cameraNum, pointDim, ptr);
      futures[i][j] = thrust::async::reduce(
          thrust::cuda::par.on(nullptr), thrust::device_ptr<T>{ptr},
          thrust::device_ptr<T>{ptr} + nItem, T(0.), thrust::plus<T>{});
      Jdx[i].push_back(ptr);
    }
  }
  for (int i = 0; i < futures.size(); ++i) {
    for (int j = futures[i].size() - 1; j >= 0; --j) {
      rhoDenominator += futures[i][j].get();
      MemoryPool::deallocateNormal(reinterpret_cast<void *>(Jdx[i][j]), i);
    }
  }
  return rhoDenominator;
}

template struct SchurLMLinearSystemManager<double>;
template struct SchurLMLinearSystemManager<float>;
}