/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include <thrust/async/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/inner_product.h>

#include <chrono>
#include <iostream>

#include "algo/lm_algo.h"
#include "edge/base_edge.h"
#include "linear_system/LM_linear_system.h"
#include "operator/jet_vector.h"
#include "resource/handle_manager.h"
#include "resource/memory_pool.h"
#include "wrapper.hpp"

namespace MegBA {
namespace {
template <typename T>
double computeResidualNorm(const JVD<T> &JV) {
  double residualNormNew = 0.;
  std::vector<std::vector<T>> residualNormNewInFlight;
  residualNormNewInFlight.resize(MemoryPool::getWorldSize());
  for (auto &vec : residualNormNewInFlight) vec.resize(JV.size());
  const auto &cublasHandle = HandleManager::getCUBLASHandle();
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
  return residualNormNew;
}

template <typename T>
inline T l2NormPow2(const T *vector, const std::size_t size) {
  return thrust::inner_product(thrust::device_ptr<const T>(vector),
                               thrust::device_ptr<const T>{vector + size},
                               thrust::device_ptr<const T>(vector), T(0.));
}

template <typename T>
__global__ void JdxpF(const T *grad, const T *deltaX, const T *res,
                      const int *absCameraPosition, const int *absPointPosition,
                      const int nItem, const int cameraDim, const int cameraNum,
                      const int pointDim, T *out) {
  unsigned int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid >= nItem) return;
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
double computeRhoDenominator(const JVD<T> &JV,
                             const BaseLinearSystem<T> &linearSystem,
                             const EdgeVector<T> &edges) {
  T rhoDenominator{0};
  std::vector<std::vector<T *>> Jdx;
  Jdx.resize(MemoryPool::getWorldSize());
  const int cameraDim = linearSystem.dim[0];
  const int cameraNum = linearSystem.num[0];
  const int pointDim = linearSystem.dim[1];

  std::vector<std::vector<thrust::system::cuda::unique_eager_future<T>>>
      futures;
  futures.resize(MemoryPool::getWorldSize());

  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = MemoryPool::getItemNum(i);
    const auto &positionContainer = edges.getPositionContainers()[i];
    futures[i].resize(JV.size());
    for (int j = 0; j < JV.size(); ++j) {
      auto &J = JV(j);
      T *ptr;
      MemoryPool::allocateNormal(reinterpret_cast<void **>(&ptr),
                                 nItem * sizeof(T), i);
      dim3 block(std::min((std::size_t)256, nItem));
      dim3 grid((nItem - 1) / block.x + 1);
      JdxpF<<<grid, block>>>(J.getCUDAGradPtr()[i], linearSystem.deltaXPtr[i],
                             J.getCUDAResPtr()[i],
                             positionContainer.absolutePosition[0],
                             positionContainer.absolutePosition[1], nItem,
                             cameraDim, cameraNum, pointDim, ptr);
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

template <typename T>
inline T linfNorm(const T *vector, const std::size_t size) {
  return std::abs(
      *thrust::max_element(thrust::device_ptr<const T>{vector},
                           thrust::device_ptr<const T>{vector + size},
                           [] __device__ __host__(T lhs, T rhs) {
                             return std::abs(lhs) < std::abs(rhs);
                           }));
}
}  // namespace
template <typename T>
void LMAlgo<T>::solveCUDA(const BaseLinearSystem<T> &baseLinearSystem,
                          const EdgeVector<T> &edges, T *xPtr) {
  auto startTimePoint = std::chrono::system_clock::now();
  const auto &linearSystem =
      dynamic_cast<const LMLinearSystem<T> &>(baseLinearSystem);
  JVD<T> jvBackup;
  jvBackup = edges.forward();
  MemoryPool::redistribute();
  edges.buildLinearSystem(jvBackup, linearSystem);
  double residualNorm, residualNormNew = computeResidualNorm(jvBackup);
  std::cout << "Start with error: " << residualNormNew / 2
            << ", log error: " << std::log10(residualNormNew / 2);

  bool stop{false};
  int k = 0;
  double v = 2.;
  linearSystem.backup();
  edges.backup();
  std::cout << ", elapsed "
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now() - startTimePoint)
                   .count()
            << " ms";
  std::cout << std::endl;
  while (!stop && k < this->algoOption.algoOptionLM.maxIter) {
    k++;
    linearSystem.processDiag(this->algoStatus.algoStatusLM);
    linearSystem.solve();
    MemoryPool::redistribute();
    cudaSetDevice(0);
    double deltaXL2 = std::sqrt(
        l2NormPow2(linearSystem.deltaXPtr[0], linearSystem.getHessianShape()));
    ;
    double xL2 = std::sqrt(l2NormPow2(xPtr, linearSystem.getHessianShape()));
    if (deltaXL2 <= this->algoOption.algoOptionLM.epsilon2 *
                        (xL2 + this->algoOption.algoOptionLM.epsilon1)) {
      break;
    }
    edges.update(linearSystem);
    double rhoDenominator =
        computeRhoDenominator(jvBackup, linearSystem, edges) - residualNormNew;
    residualNorm = residualNormNew;
    JVD<T> jv = edges.forward();
    residualNormNew = computeResidualNorm(jv);
    double rho = -(residualNorm - residualNormNew) / rhoDenominator;
    if (residualNorm > residualNormNew) {
      jvBackup = jv;
      edges.buildLinearSystem(jv, linearSystem);
      std::cout << "Iter " << k << " error: " << residualNormNew / 2
                << ", log error: " << std::log10(residualNormNew / 2);
      linearSystem.backup();
      edges.backup();
      linearSystem.applyUpdate(xPtr);

      residualNorm = residualNormNew;
      this->algoStatus.algoStatusLM.region /=
          std::max(1. / 3., 1 - std::pow(2 * rho - 1, 3));
      v = 2.;
      this->algoStatus.algoStatusLM.recoverDiag = false;

      cudaSetDevice(0);
      const auto norm =
          linfNorm(linearSystem.g[0], linearSystem.getHessianShape());
      stop = norm <= this->algoOption.algoOptionLM.epsilon1;
    } else {
      std::cout << "Iter " << k << " failed";
      linearSystem.rollback();
      edges.rollback();
      residualNormNew = residualNorm;
      this->algoStatus.algoStatusLM.region /= v;
      v *= 2;
      this->algoStatus.algoStatusLM.recoverDiag = true;
    }
    std::cout << ", elapsed "
              << std::chrono::duration_cast<std::chrono::milliseconds>(
                     std::chrono::system_clock::now() - startTimePoint)
                     .count()
              << " ms";
    std::cout << std::endl;
  }
  std::cout << "Finished" << std::endl;
}

SPECIALIZE_CLASS(LMAlgo);
}  // namespace MegBA
