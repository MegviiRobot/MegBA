/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/base_problem.h"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <thrust/async/reduce.h>
#include "wrapper.hpp"
#include "resource/handle_manager.h"
#include "macro.h"
#include "solver/base_solver.h"
#include "linear_system_manager/schurLM_linear_system_manager.h"

namespace MegBA {
template <typename T> void BaseProblem<T>::deallocateResourceCUDA() {
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaFree(xPtr[i]);
      cudaFree(deltaXPtr[i]);
      cudaFree(deltaXPtrBackup[i]);
    }
    xPtr.clear();
    deltaXPtr.clear();
    deltaXPtrBackup.clear();
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T> void BaseProblem<T>::prepareUpdateDataCUDA() {
  if (option.useSchur) {
    const auto worldSize = MemoryPool::getWorldSize();
    xPtr.resize(worldSize);
    deltaXPtr.resize(worldSize);
    deltaXPtrBackup.resize(worldSize);
    for (int i = 0; i < worldSize; ++i) {
      cudaSetDevice(i);
      cudaMalloc(&xPtr[i], hessianShape * sizeof(T));
      cudaMalloc(&deltaXPtr[i], hessianShape * sizeof(T));
      cudaMalloc(&deltaXPtrBackup[i], hessianShape * sizeof(T));
      cudaMemsetAsync(deltaXPtr[i], 0, hessianShape * sizeof(T));
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

namespace {
template <typename T> struct compare_abs_value {
  __host__ __device__ bool operator()(T lhs, T rhs) {
    return std::abs(lhs) < std::abs(rhs);
  }
};

template <typename T> inline T l2NormPow2(const T *vector, const int size) {
  return thrust::inner_product(thrust::device_ptr<const T>(vector),
                               thrust::device_ptr<const T>{vector + size},
                               thrust::device_ptr<const T>(vector), T(0.));
}

template <typename T> inline T LinfNorm(const T *vector, const int size) {
  return std::abs(*thrust::max_element(
      thrust::device_ptr<const T>{vector},
      thrust::device_ptr<const T>{vector + size}, compare_abs_value<T>{}));
}

template <typename T>
__global__ void ExtractOldAndApplyNewDiagKernel(const T a, const int batchSize,
                                                T *csrVal, T *diags) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize)
    return;

  const T diag = csrVal[threadIdx.x + threadIdx.x * blockDim.x +
                        tid * blockDim.x * blockDim.x];
  diags[threadIdx.x + tid * blockDim.x] = diag;
  csrVal[threadIdx.x + threadIdx.x * blockDim.x +
         tid * blockDim.x * blockDim.x] = (a + 1) * diag;
}

template <typename T>
__global__ void RecoverDiagKernel(const T *in, const T a, const int batchSize,
                                  T *out) {
  /*
   * blockDim, x-dim: camera or point dim, y-dim: process how many cameras/points in this block
   */
  unsigned int tid = threadIdx.y + blockIdx.x * blockDim.y;
  if (tid >= batchSize)
    return;

  out[threadIdx.x + threadIdx.x * blockDim.x + tid * blockDim.x * blockDim.x] =
      (a + 1) * in[threadIdx.x + tid * blockDim.x];
}

template <typename T>
void extractOldAndApplyNewDiag(const T a, const int batchSize, const int dim,
                               T *csrVal, T *diag) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  ExtractOldAndApplyNewDiagKernel<<<grid, block>>>(a, batchSize, csrVal, diag);
}

template <typename T>
void RecoverDiag(const T *diag, const T a, const int batchSize, const int dim,
                 T *csrVal) {
  dim3 block(dim, std::min(decltype(batchSize)(32), batchSize));
  dim3 grid((batchSize - 1) / block.y + 1);
  RecoverDiagKernel<T><<<grid, block>>>(diag, a, batchSize, csrVal);
}

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

template <typename T>
double computeRhoDenominator(JVD<T> &JV, std::vector<T *> &schurDeltaXPtr, EdgeVector<T> &edges) {
  T rhoDenominator{0};
  std::vector<std::vector<T *>> Jdx;
  Jdx.resize(MemoryPool::getWorldSize());
  const int cameraDim = edges.schurEquationContainer[0].dim[0];
  const int cameraNum =
      edges.schurEquationContainer[0].nnz[2] / cameraDim / cameraDim;
  const int pointDim = edges.schurEquationContainer[0].dim[1];

  std::vector<std::vector<thrust::system::cuda::unique_eager_future<T>>>
      futures;
  futures.resize(MemoryPool::getWorldSize());

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = MemoryPool::getItemNum(i);
    const auto &positionContainer =
        edges.schurPositionAndRelationContainer[i];
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
}  // namespace

template <typename T>
void BaseProblem<T>::solveLM() {
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  makeVertices();
  Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic> JV_backup;
  int k = 0;
  T residualNormNew = 0;
  T residualNorm = 0;

  edges.backupValueDevicePtrs();
  edges.bindCUDAGradPtrs();
  JV_backup = edges.forward();
  if (option.useSchur) {
    edges.buildLinearSystemSchur(JV_backup);
  } else {
    // TODO(Jie Ren): implement this
  }

  std::vector<std::vector<T>> residualNormNewInFlight;
  residualNormNewInFlight.resize(MemoryPool::getWorldSize());
  for (auto &vec : residualNormNewInFlight)
    vec.resize(JV_backup.size());
  for (int i = 0; i < JV_backup.rows(); ++i) {
    for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
      cudaSetDevice(j);
      const T *resPtr = JV_backup(i).getCUDAResPtr()[j];
      Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getItemNum(j),
                                resPtr, 1, resPtr, 1,
                                &residualNormNewInFlight[j][i]);
    }
  }
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaStream_t stream;
    cudaSetDevice(i);
    cublasGetStream_v2(cublasHandle[i], &stream);
    cudaStreamSynchronize(stream);
    for (const auto residualNormNewLanded : residualNormNewInFlight[i]) {
      residualNormNew += residualNormNewLanded;
    }
  }

  std::cout << "start with error: " << residualNormNew / 2
            << ", log error: " << std::log10(residualNormNew / 2) << std::endl;

  MemoryPool::redistribute();
  bool stop{false};
  T u = option.algoOptionLM.initialRegion;
  T v = 2;
  T rho = 0;

  std::vector<std::array<T *, 2>> extractedDiag;
  extractedDiag.resize(MemoryPool::getWorldSize());
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    auto &container = edges.schurEquationContainer[i];
    cudaMalloc(&extractedDiag[i][0],
               container.nnz[2] / container.dim[0] * sizeof(T));
    cudaMalloc(&extractedDiag[i][1],
               container.nnz[3] / container.dim[1] * sizeof(T));
  }
  bool recoverDiagFlag{false};
  while (!stop && k < option.algoOptionLM.maxIter) {
    k++;
    if (option.useSchur) {
      if (recoverDiagFlag) {
        for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
          cudaSetDevice(i);
          auto &container = edges.schurEquationContainer[i];
          ASSERT_CUDA_NO_ERROR();
          RecoverDiag(extractedDiag[i][0], T(1.) / u,
                      container.nnz[2] / container.dim[0] / container.dim[0],
                      container.dim[0], container.csrVal[2]);
          ASSERT_CUDA_NO_ERROR();
          RecoverDiag(extractedDiag[i][1], T(1.) / u,
                      container.nnz[3] / container.dim[1] / container.dim[1],
                      container.dim[1], container.csrVal[3]);
          ASSERT_CUDA_NO_ERROR();
        }
        recoverDiagFlag = false;
      } else {
        for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
          cudaSetDevice(i);
          auto &container = edges.schurEquationContainer[i];
          extractOldAndApplyNewDiag(
              T(1.) / u, container.nnz[2] / container.dim[0] / container.dim[0],
              container.dim[0], container.csrVal[2], extractedDiag[i][0]);
          extractOldAndApplyNewDiag(
              T(1.) / u, container.nnz[3] / container.dim[1] / container.dim[1],
              container.dim[1], container.csrVal[3], extractedDiag[i][1]);
        }
      }
    } else {
      // TODO(Jie Ren): implement this
    }
//    bool solverSuccess = solveLinear();
    solver->solve();
    MemoryPool::redistribute();
    ASSERT_CUDA_NO_ERROR();

    T deltaXL2, xL2;
    if (option.useSchur) {
      cudaSetDevice(0);
      deltaXL2 = l2NormPow2(deltaXPtr[0], hessianShape);
      xL2 = l2NormPow2(xPtr[0], hessianShape);
    } else {
      // TODO(Jie Ren): implement this
    }
    ASSERT_CUDA_NO_ERROR();

    deltaXL2 = std::sqrt(deltaXL2);
    xL2 = std::sqrt(xL2);
    if (deltaXL2 <= option.algoOptionLM.epsilon2 * (xL2 + option.algoOptionLM.epsilon1)) {
      break;
    } else {
      if (option.useSchur) {
        edges.updateSchur(deltaXPtr);
      } else {
        // TODO(Jie Ren): implement this
      }

      T rhoDenominator{0};
      if (option.useSchur) {
        rhoDenominator = computeRhoDenominator(JV_backup, deltaXPtr, edges);
        rhoDenominator -= residualNormNew;
      } else {
        // TODO(Jie Ren): implement this
      }
      ASSERT_CUDA_NO_ERROR();

      residualNorm = residualNormNew;
      residualNormNew = 0.;
      auto JV = edges.forward();
      for (int i = 0; i < JV.size(); ++i) {
        for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
          cudaSetDevice(j);
          const T *resPtr = JV(i).getCUDAResPtr()[j];
          Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getItemNum(j),
                                    resPtr, 1, resPtr, 1,
                                    &residualNormNewInFlight[j][i]);
        }
      }
      for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
        cudaStream_t stream;
        cudaSetDevice(i);
        cublasGetStream_v2(cublasHandle[i], &stream);
        cudaStreamSynchronize(stream);
        for (const auto residualNormNewLanded : residualNormNewInFlight[i]) {
          residualNormNew += residualNormNewLanded;
        }
      }
      ASSERT_CUDA_NO_ERROR();

      rho = -(residualNorm - residualNormNew) / rhoDenominator;

      if (residualNorm > residualNormNew) {
        for (int i = 0; i < JV.size(); ++i)
          JV_backup(i) = JV(i);
        if (option.useSchur) {
          edges.buildLinearSystemSchur(JV);
        } else {
          // TODO(Jie Ren): implement this
        }
        std::cout << k << "-th iter error: " << residualNormNew / 2
                  << ", log error: " << std::log10(residualNormNew / 2)
                  << std::endl;

        backupLM();
        residualNorm = residualNormNew;
        if (option.useSchur) {
          cudaSetDevice(0);
          auto &container = edges.schurEquationContainer[0];
          const auto norm = LinfNorm(container.g, hessianShape);
          stop = norm <= option.algoOptionLM.epsilon1;
        } else {
          // TODO(Jie Ren): implement this
        }
        u /= std::max(1. / 3., 1 - std::pow(2 * rho - 1, 3));
        v = 2;
      } else {
        residualNormNew = residualNorm;
        rollbackLM();
        u /= v;
        v *= 2;
        recoverDiagFlag = true;
      }
    }
    if (stop)
      break;
  }
  writeBack();
  deallocateResource();
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaFree(extractedDiag[i][0]);
    cudaFree(extractedDiag[i][1]);
  }
}

template <typename T> void BaseProblem<T>::backupLM() {
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
  T one = 1.;
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(deltaXPtrBackup[i], deltaXPtr[i],
                      hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
      Wrapper::cublasGaxpy::call(cublasHandle[i], hessianShape, &one,
                                 deltaXPtr[i], 1, xPtr[i], 1);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
  edges.backupValueDevicePtrs();
}

template <typename T> void BaseProblem<T>::rollbackLM() {
  edges.rollback();
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(deltaXPtr[i], deltaXPtrBackup[i],
                      hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
