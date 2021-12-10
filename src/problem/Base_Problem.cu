/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "problem/BaseProblem.h"
#include "Wrapper.hpp"
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>
#include <resource/Manager.h>
#include <Macro.h>
#include <thrust/async/reduce.h>

namespace MegBA {
template <typename T> void BaseProblem<T>::deallocateResourceCUDA() {
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaFree(schurXPtr[i]);
      cudaFree(schurDeltaXPtr[i]);
      cudaFree(schurDeltaXPtrBackup[i]);
    }
    schurXPtr.clear();
    schurDeltaXPtr.clear();
    schurDeltaXPtrBackup.clear();
  } else {
    // TODO: implement this
  }
}

template <typename T> void BaseProblem<T>::prepareUpdateDataCUDA() {
  if (option.useSchur) {
    const auto world_size = MemoryPool::getWorldSize();
    schurXPtr.resize(world_size);
    schurDeltaXPtr.resize(world_size);
    schurDeltaXPtrBackup.resize(world_size);
    for (int i = 0; i < world_size; ++i) {
      cudaSetDevice(i);
      cudaMalloc(&schurXPtr[i], hessianShape * sizeof(T));
      cudaMalloc(&schurDeltaXPtr[i], hessianShape * sizeof(T));
      cudaMalloc(&schurDeltaXPtrBackup[i], hessianShape * sizeof(T));
      cudaMemsetAsync(schurDeltaXPtr[i], 0, hessianShape * sizeof(T));
    }
  } else {
    // TODO: implement this
  }
}

namespace {
template <typename T> struct compare_abs_value {
  __host__ __device__ bool operator()(T lhs, T rhs) {
    return std::abs(lhs) < std::abs(rhs);
  }
};

template <typename T>
struct compare_abs_value_ret_max : public thrust::binary_function<T, T, T> {
  __host__ __device__ T operator()(T lhs, T rhs) {
    return std::abs(lhs) < std::abs(rhs) ? std::abs(rhs) : std::abs(lhs);
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
inline auto LinfNorm_async(const T *vector, const int size) {
  return thrust::async::reduce(thrust::device_ptr<const T>{vector},
                               thrust::device_ptr<const T>{vector + size},
                               T(0.), compare_abs_value_ret_max<T>{});
}

namespace {
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
                      const int *absCameraPosition,
                      const int *absPointPosition, const int nElm,
                      const int cameraDim, const int cameraNum,
                      const int pointDim, T *out) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nElm)
    return;
  T sum{0};
  const int absCameraPositionLocal = absCameraPosition[tid];
  const int absPointPositionLocal = absPointPosition[tid];
  for (int i = 0; i < cameraDim; ++i) {
    sum += grad[tid + i * nElm] *
           deltaX[i + absCameraPositionLocal * cameraDim];
  }
  for (int i = 0; i < pointDim; ++i) {
    sum += grad[tid + (i + cameraDim) * nElm] *
        deltaX[i + cameraDim * cameraNum + absPointPositionLocal * pointDim];
  }
  out[tid] = (sum + res[tid]) * (sum + res[tid]);
}
}
    template <typename T>
void BaseProblem<T>::solveLM(int iter, double solverTol,
                             double solverRefuseRatio, int solverMaxIter,
                             const double tau, const double epsilon1,
                             const double epsilon2) {
  const auto &cublasHandle = HandleManager::get_cublasHandle();
  makeVertices();
  Eigen::Matrix<JetVector<T>, Eigen::Dynamic, Eigen::Dynamic> JV_backup;
  int k = 0;
  T residualNormNew = 0;
  T residualNorm = 0;

  edges.backupDaPtrs();
  edges.regetCUDAGradPtrs();
  JV_backup = edges.forward();
  if (option.useSchur) {
    edges.buildLinearSystemSchur(JV_backup);
  } else {
    // TODO: implement this
  }

  std::vector<std::vector<T>> residualNormNewInFlight;
  residualNormNewInFlight.resize(MemoryPool::getWorldSize());
  for (auto &vec : residualNormNewInFlight)
    vec.resize(JV_backup.size());
  for (int i = 0; i < JV_backup.rows(); ++i) {
    for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
      cudaSetDevice(j);
      const T *resPtr = JV_backup(i).getCUDAResPtr()[j];
      Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getElmNum(j),
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
            << ", log error: " << std::log10(residualNormNew / 2)
            << std::endl;

  MemoryPool::redistribute();
  bool stop{false};
  T u = tau;
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
  while (!stop && k < iter) {
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
      // TODO: implement this
    }
    bool solverSuccess =
        solveLinear(solverTol, solverRefuseRatio, solverMaxIter);
    MemoryPool::redistribute();
    ASSERT_CUDA_NO_ERROR();

    T deltaXL2, xL2;
    if (option.useSchur) {
      cudaSetDevice(0);
      deltaXL2 = l2NormPow2(schurDeltaXPtr[0], hessianShape);
      xL2 = l2NormPow2(schurXPtr[0], hessianShape);
    } else {
      // TODO: implement this
    }
    ASSERT_CUDA_NO_ERROR();

    deltaXL2 = std::sqrt(deltaXL2);
    xL2 = std::sqrt(xL2);
    if (deltaXL2 <= epsilon2 * (xL2 + epsilon1)) {
      std::cout << "Stopped for deltaXL2{" << deltaXL2 << "} <= epsilon2{"
                << epsilon2 << "} * (xL2{" << xL2 << "} + epsilon1{"
                << epsilon1 << "})" << std::endl;
      break;
    } else {
      if (option.useSchur) {
        edges.updateSchur(schurDeltaXPtr);
      } else {
        // TODO: implement this
      }

      T rhoDenominator{0};
      if (option.useSchur) {
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
          const auto nElm = MemoryPool::getElmNum(i);
          const auto &eqContainer = edges.schurEquationContainer[i];
          const auto &positionContainer =
              edges.schurPositionAndRelationContainer[i];
          futures[i].resize(JV_backup.size());
          for (int j = 0; j < JV_backup.size(); ++j) {
            auto &J = JV_backup(j);
            T *ptr;
            MemoryPool::allocateNormal((void **)&ptr, nElm * sizeof(T), i);
            dim3 block(std::min((std::size_t)256, nElm));
            dim3 grid((nElm - 1) / block.x + 1);
            JdxpF<<<grid, block>>>(
                J.getCUDAGradPtr()[i], schurDeltaXPtr[i],
                J.getCUDAResPtr()[i],
                                   positionContainer.absolutePositionCamera,
                                   positionContainer.absolutePositionPoint, nElm, cameraDim, cameraNum, pointDim, ptr);
            futures[i][j] = thrust::async::reduce(
                thrust::cuda::par.on(nullptr), thrust::device_ptr<T>{ptr},
                thrust::device_ptr<T>{ptr} + nElm, T(0.), thrust::plus<T>{});
            Jdx[i].push_back(ptr);
          }
        }
        for (int i = 0; i < futures.size(); ++i) {
          for (int j = futures[i].size() - 1; j >= 0; --j) {
            rhoDenominator += futures[i][j].get();
            MemoryPool::deallocateNormal((void *)Jdx[i][j], i);
          }
        }
        rhoDenominator -= residualNormNew;
      } else {
        // TODO: implement this
      }
      ASSERT_CUDA_NO_ERROR();

      residualNorm = residualNormNew;
      residualNormNew = 0.;
      auto JV = edges.forward();
      for (int i = 0; i < JV.size(); ++i) {
        for (int j = 0; j < MemoryPool::getWorldSize(); ++j) {
          cudaSetDevice(j);
          const T *resPtr = JV(i).getCUDAResPtr()[j];
          Wrapper::cublasGdot::call(cublasHandle[j], MemoryPool::getElmNum(j),
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
          // TODO: implement this
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
          stop = norm <= epsilon1;
          if (stop)
            std::cout << "Stopped for norm{" << norm << "} <= epsilon1{"
                      << epsilon1 << "}" << std::endl;
        } else {
          // TODO: implement this
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
  const std::vector<cublasHandle_t> &cublasHandle =
      HandleManager::get_cublasHandle();
  T one = 1.;
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurDeltaXPtrBackup[i], schurDeltaXPtr[i],
                      hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
      Wrapper::cublasGaxpy::call(cublasHandle[i], hessianShape, &one,
                                 schurDeltaXPtr[i], 1, schurXPtr[i], 1);
    }
  } else {
    // TODO: implement this
  }
  edges.backupDaPtrs();
}

template <typename T> void BaseProblem<T>::rollbackLM() {
  edges.rollback();
  if (option.useSchur) {
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurDeltaXPtr[i], schurDeltaXPtrBackup[i],
                      hessianShape * sizeof(T), cudaMemcpyDeviceToDevice);
    }
  } else {
    // TODO: implement this
  }
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}