/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>

#include "edge/base_edge.h"
#include "linear_system/schur_LM_linear_system.h"
#include "macro.h"
#include "resource/handle_manager.h"
#include "wrapper.hpp"

#if __CUDA_ARCH__ < 600 && defined(__CUDA_ARCH__)
namespace {
union AtomicUnion {
  double dValue;
  unsigned long long ullValue;
};
}  // namespace

__inline__ __device__ double atomicAdd(double *address, double val) {
  AtomicUnion old, assumed;
  old.dValue = *address;

  do {
    assumed = old;
    old.ullValue =
        atomicCAS(reinterpret_cast<unsigned long long *>(address),
                  assumed.ullValue, AtomicUnion{val + assumed.dValue}.ullValue);

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
   * make sure that blockDim.x % 32 == 0, if so, there won't be any thread
   * divergence within a wrap.
   */
  const unsigned int tid = threadIdx.x + blockDim.x * blockIdx.x;
  if (tid >= errorNum) return;

  T *valSmem = Wrapper::SharedMemory<T>::get();

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
                  (threadIdx.y - cameraDim) * pointDim,
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
void EdgeVector<T>::buildLinearSystemCUDA(
    const JVD<T> &jetEstimation,
    const BaseLinearSystem<T> &linearSystem) const {
  const auto &linearSystemLocal =
      dynamic_cast<const SchurLMLinearSystem<T> &>(linearSystem);
  const auto rows = jetEstimation.rows(), cols = jetEstimation.cols();
  const auto cameraDim = linearSystemLocal.dim[0];
  const auto pointDim = linearSystemLocal.dim[1];
  const auto cameraNum = linearSystemLocal.num[0];
  const auto pointNum = linearSystemLocal.num[1];
  const auto hppRows = cameraDim * cameraNum;
  const auto hllRows = pointDim * pointNum;
  ASSERT_CUDA_NO_ERROR();

  std::vector<T *> gCameraDevice{option.deviceUsed.size()};
  std::vector<T *> gPointDevice{option.deviceUsed.size()};
  for (int i = 0; i < option.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    cudaMemsetAsync(linearSystemLocal.g[i], 0, (hppRows + hllRows) * sizeof(T));
    gCameraDevice[i] = &linearSystemLocal.g[i][0];
    gPointDevice[i] = &linearSystemLocal.g[i][hppRows];
    ASSERT_CUDA_NO_ERROR();
    cudaMemsetAsync(linearSystemLocal.equationContainers[i].csrVal[0], 0,
                    linearSystemLocal.equationContainers[i].nnz[0] * sizeof(T));
    cudaMemsetAsync(linearSystemLocal.equationContainers[i].csrVal[1], 0,
                    linearSystemLocal.equationContainers[i].nnz[1] * sizeof(T));
    cudaMemsetAsync(linearSystemLocal.equationContainers[i].csrVal[2], 0,
                    linearSystemLocal.equationContainers[i].nnz[2] * sizeof(T));
    cudaMemsetAsync(linearSystemLocal.equationContainers[i].csrVal[3], 0,
                    linearSystemLocal.equationContainers[i].nnz[3] * sizeof(T));
    ASSERT_CUDA_NO_ERROR();
  }
  ASSERT_CUDA_NO_ERROR();

  const auto resDim = rows * cols;
  std::vector<std::unique_ptr<const T *[]>> totalPtrs {};
  totalPtrs.reserve(option.deviceUsed.size());
  std::vector<const T **> totalPtrsDevice{option.deviceUsed.size()};

  std::vector<const T **> valPtrs{option.deviceUsed.size()};
  std::vector<const T **> valPtrsDevice{option.deviceUsed.size()};

  std::vector<const T **> errorPtrs{option.deviceUsed.size()};
  std::vector<const T **> errorPtrsDevice{option.deviceUsed.size()};
  for (int deviceRank = 0; deviceRank < option.deviceUsed.size();
       ++deviceRank) {
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
    ASSERT_CUDA_NO_ERROR();
    for (int i = 0; i < option.deviceUsed.size(); ++i) {
      ASSERT_CUDA_NO_ERROR();
      cudaSetDevice(i);
      const auto edgeNum = MemoryPool::getItemNum(i);
      dim3 block(std::min((decltype(edgeNum))32, edgeNum),
                 cameraDim + pointDim);
      dim3 grid((edgeNum - 1) / block.x + 1);
      makeHSchur<<<grid, block, block.x * block.y * sizeof(T)>>>(
          valPtrsDevice[i], errorPtrsDevice[i],
          positionContainers[i].absolutePosition[0],
          positionContainers[i].absolutePosition[1],
          positionContainers[i].relativePosition[0],
          positionContainers[i].relativePosition[1],
          linearSystemLocal.equationContainers[i].csrRowPtr[0],
          linearSystemLocal.equationContainers[i].csrRowPtr[1], resDim,
          cameraDim, pointDim, edgeNum, gCameraDevice[i], gPointDevice[i],
          linearSystemLocal.equationContainers[i].csrVal[2],
          linearSystemLocal.equationContainers[i].csrVal[3],
          linearSystemLocal.equationContainers[i].csrVal[0],
          linearSystemLocal.equationContainers[i].csrVal[1]);
    }
  }
  ASSERT_CUDA_NO_ERROR();
  for (int i = 0; i < option.deviceUsed.size(); ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(nullptr);
    cudaFree(totalPtrsDevice[i]);
  }

  const auto &comms = HandleManager::getNCCLComm();
  ncclGroupStart();
  for (int i = 0; i < option.deviceUsed.size(); ++i) {
    ncclAllReduce(linearSystemLocal.equationContainers[i].csrVal[2],
                  linearSystemLocal.equationContainers[i].csrVal[2],
                  linearSystemLocal.equationContainers[i].nnz[2],
                  Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                  nullptr);
    ncclAllReduce(linearSystemLocal.equationContainers[i].csrVal[3],
                  linearSystemLocal.equationContainers[i].csrVal[3],
                  linearSystemLocal.equationContainers[i].nnz[3],
                  Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                  nullptr);
    ncclAllReduce(gCameraDevice[i], gCameraDevice[i], hppRows + hllRows,
                  Wrapper::declaredDtype<T>::ncclDtype, ncclSum, comms[i],
                  nullptr);
  }
  ncclGroupEnd();
}

template class EdgeVector<double>;
template class EdgeVector<float>;
}  // namespace MegBA
