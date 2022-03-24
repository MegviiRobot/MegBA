/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "edge/base_edge.h"
#include "linear_system/base_linear_system.h"

namespace MegBA {
namespace {
template <typename T>
__global__ void updateDeltaXTwoVertices(const T *deltaX,
                                        const int *absolutePositionCamera,
                                        const int *absolutePositionPoint,
                                        const int cameraDim, const int pointDim,
                                        const int cameraNum, const int nItem,
                                        T *cameraX, T *pointX) {
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nItem) return;
  // ix : index in edges
  // absolute_position_camera[ix] :
  // index in cameras
  // absolute_position_camera[ix] * camera_dim :
  // starting position of its camera_block(dim = camera_dim)
  // threadIdx.y :
  // offset of its camera_block
  unsigned int idx = tid;
  for (int i = 0; i < cameraDim; ++i) {
    cameraX[idx] += deltaX[absolutePositionCamera[tid] * cameraDim + i];
    idx += nItem;
  }

  for (int i = 0; i < pointDim; ++i) {
    pointX[idx - nItem * cameraDim] +=
        deltaX[absolutePositionPoint[tid] * pointDim + i +
               cameraNum * cameraDim];
    idx += nItem;
  }
}
}  // namespace

template <typename T>
void EdgeVector<T>::update(const BaseLinearSystem<T> &linearSystem) const {
  const auto cameraDim = linearSystem.dim[0];
  const auto cameraNum = linearSystem.num[0];
  const auto pointDim = linearSystem.dim[1];

  // TODO(Jie Ren): merge into method 'solve_Linear'
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nItem = MemoryPool::getItemNum(i);
    dim3 block(std::min((decltype(nItem))256, nItem));
    dim3 grid((nItem - 1) / block.x + 1);
    updateDeltaXTwoVertices<T><<<grid, block>>>(
        linearSystem.deltaXPtr[i], positionContainers[i].absolutePosition[0],
        positionContainers[i].absolutePosition[1], cameraDim, pointDim,
        cameraNum, nItem, schurValueDevicePtrs[0][i],
        schurValueDevicePtrs[1][i]);
  }
}

template <typename T>
void EdgeVector<T>::bindCUDAGradPtrs() {
  int vertexKindIdxUnfixed = 0;
  for (auto &vertexVector : edges) {
    if (vertexVector[0]->fixed) continue;
    auto &jetEstimation = vertexVector.getJVEstimation();
    auto &jetObservation = vertexVector.getJVObservation();

    const auto worldSize = MemoryPool::getWorldSize();
    for (int i = 0; i < vertexVector[0]->getEstimation().size(); ++i) {
      // bind _valueDevicePtr for CUDA
      if (option.useSchur) {
        std::vector<T *> valueDevicePtrs;
        valueDevicePtrs.resize(worldSize);
        for (int k = 0; k < worldSize; ++k) {
          valueDevicePtrs[k] =
              &schurValueDevicePtrs[vertexKindIdxUnfixed][k]
                                   [i * MemoryPool::getItemNum(k)];
        }
        jetEstimation(i).bindValueDevicePtr(std::move(valueDevicePtrs));
      } else {
        // TODO(Jie Ren): implement this
      }
    }
    vertexKindIdxUnfixed++;
  }
}
template class EdgeVector<double>;
template class EdgeVector<float>;
}  // namespace MegBA
