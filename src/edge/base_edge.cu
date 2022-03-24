/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "edge/base_edge.h"
#include "macro.h"

namespace MegBA {
namespace {
void CUDART_CB freeCallback(void *ptr) { free(ptr); }
}  // namespace

template <typename T>
void EdgeVector<T>::backup() const {
  if (option.useSchur) {
    const auto gradShape = getGradShape();
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurValueDevicePtrsOld[0][i], schurValueDevicePtrs[0][i],
                      MemoryPool::getItemNum(i) * gradShape * sizeof(T),
                      cudaMemcpyDeviceToDevice);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T>
void EdgeVector<T>::rollback() const {
  if (option.useSchur) {
    const auto gradShape = getGradShape();
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaMemcpyAsync(schurValueDevicePtrs[0][i], schurValueDevicePtrsOld[0][i],
                      MemoryPool::getItemNum(i) * gradShape * sizeof(T),
                      cudaMemcpyDeviceToDevice);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T>
void EdgeVector<T>::allocateResourceCUDA() {
  if (option.useSchur) {
    const auto worldSize = MemoryPool::getWorldSize();
    const auto gradShape = getGradShape();
    std::vector<T *> valueDevicePtrs, valueDevicePtrsOld;
    valueDevicePtrs.resize(worldSize);
    valueDevicePtrsOld.resize(worldSize);
    for (int i = 0; i < worldSize; ++i) {
      const auto edgeNum = MemoryPool::getItemNum(i);
      cudaSetDevice(i);
      T *valueDevicePtr, *valueDevicePtrOld;
      const auto nItem = MemoryPool::getItemNum(i);
      cudaMalloc(&valueDevicePtr, nItem * gradShape * sizeof(T));
      cudaMalloc(&valueDevicePtrOld, nItem * gradShape * sizeof(T));
      valueDevicePtrs[i] = valueDevicePtr;
      valueDevicePtrsOld[i] = valueDevicePtrOld;

      for (int j = 0; j < cameraVertexNum + pointVertexNum; ++j) {
        int *tmpPtr = positionContainers[i].relativePosition[j];
        cudaMalloc(&positionContainers[i].relativePosition[j],
                   edgeNum * sizeof(int));

        cudaMemcpyAsync(positionContainers[i].relativePosition[j], tmpPtr,
                        edgeNum * sizeof(int), cudaMemcpyHostToDevice);
        cudaLaunchHostFunc(nullptr, freeCallback, (void *)tmpPtr);

        tmpPtr = positionContainers[i].absolutePosition[j];
        cudaMalloc(&positionContainers[i].absolutePosition[j],
                   edgeNum * sizeof(int));
        cudaMemcpyAsync(positionContainers[i].absolutePosition[j], tmpPtr,
                        edgeNum * sizeof(int), cudaMemcpyHostToDevice);
        cudaLaunchHostFunc(nullptr, freeCallback, (void *)tmpPtr);
        ASSERT_CUDA_NO_ERROR();
      }
    }
    schurValueDevicePtrs.resize(cameraVertexNum + pointVertexNum);
    schurValueDevicePtrsOld.resize(cameraVertexNum + pointVertexNum);
    for (int i = 0; i < cameraVertexNum + pointVertexNum; ++i) {
      schurValueDevicePtrs[i].resize(worldSize);
      schurValueDevicePtrsOld[i].resize(worldSize);
    }
    for (int i = 0, iUnfixed = 0, offset = 0; i < edges.size(); ++i) {
      if (edges[i][0]->fixed) continue;
      for (int j = 0; j < worldSize; ++j) {
        const auto nItem = MemoryPool::getItemNum(j);
        schurValueDevicePtrs[iUnfixed][j] = &valueDevicePtrs[j][offset * nItem];
        schurValueDevicePtrsOld[iUnfixed][j] =
            &valueDevicePtrsOld[j][offset * nItem];
      }
      iUnfixed++;
      const auto &estimation = edges[i][0]->getEstimation();
      offset += estimation.rows() * estimation.cols();
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template <typename T>
void EdgeVector<T>::deallocateResourceCUDA() {
  if (option.useSchur) {
    for (auto &edge : edges) edge.CPU();
    for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
      cudaSetDevice(i);
      cudaFree(schurValueDevicePtrs[0][i]);
      cudaFree(schurValueDevicePtrsOld[0][i]);
      for (auto p : positionContainers[i].relativePosition) cudaFree(p);
      for (auto p : positionContainers[i].absolutePosition) cudaFree(p);
    }
  } else {
    // TODO(Jie Ren): implement this
  }
}

template class EdgeVector<float>;
template class EdgeVector<double>;
}  // namespace MegBA
