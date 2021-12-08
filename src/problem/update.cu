/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "edge/BaseEdge.h"
#include <Macro.h>

namespace MegBA {
namespace {
template <typename T>
__global__ void
updateDeltaXTwoVertices(const T *deltaX, const int *absolutePositionCamera,
                        const int *absolutePositionPoint, const int cameraDim,
                        const int pointDim, const int cameraNum, const int nElm,
                        T *cameraX, T *pointX) {
  const unsigned int tid = blockDim.x * blockIdx.x + threadIdx.x;
  if (tid >= nElm)
    return;
  // ix : index in edges
  // absolute_position_camera[ix] : index in cameras
  // absolute_position_camera[ix] * camera_dim : starting position of its camera_block(dim = camera_dim) threadIdx.y : offset of its camera_block
  unsigned int idx = tid;
  for (int i = 0; i < cameraDim; ++i) {
    cameraX[idx] += deltaX[absolutePositionCamera[tid] * cameraDim + i];
    idx += nElm;
  }

  for (int i = 0; i < pointDim; ++i) {
    pointX[idx - nElm * cameraDim] +=
        deltaX[absolutePositionPoint[tid] * pointDim + i +
               cameraNum * cameraDim];
    idx += nElm;
  }
}
}

    template <typename T>
void EdgeVector<T>::updateSchur(const std::vector<T *> &deltaXPtr) {
  for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaStreamSynchronize(schurStreamLmMemcpy[i]);
  }

  const auto cameraDim = edges[0][0]->getGradShape();
  const auto cameraNum =
      verticesSetPtr->find(edges[0][0]->kind())->second.size();
  const auto pointDim = edges[1][0]->getGradShape();

  // TODO: merge into method 'solve_Linear'
  for (int i = 0; i < Memory_Pool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    const auto nElm = Memory_Pool::getElmNum(i);
    dim3 block(std::min((decltype(nElm))256, nElm));
    dim3 grid((nElm - 1) / block.x + 1);
    updateDeltaXTwoVertices<T><<<grid, block>>>(
        deltaXPtr[i],
        schurPositionAndRelationContainer[i].absolutePositionCamera,
        schurPositionAndRelationContainer[i].absolutePositionPoint, cameraDim,
        pointDim, cameraNum, nElm, schurDaPtrs[0][i], schurDaPtrs[1][i]);
  }
}

template <typename T> void EdgeVector<T>::rebindDaPtrs() {
  int vertexKindIdxUnfixed = 0;
  for (auto &vertexVector : edges) {
    if (vertexVector[0]->get_Fixed())
      continue;
    auto &jetEstimation = vertexVector.get_Jet_Estimation();
    auto &jetObservation = vertexVector.get_Jet_Observation();

    const auto worldSize = Memory_Pool::getWorldSize();
    for (int i = 0; i < vertexVector[0]->get_Estimation().size(); ++i) {
      // bind da_ptr_ for CUDA
      if (_option.useSchur) {
        std::vector<T *> daPtrs;
        daPtrs.resize(worldSize);
        for (int k = 0; k < worldSize; ++k) {
          daPtrs[k] = &schurDaPtrs[vertexKindIdxUnfixed][k]
                                  [i * Memory_Pool::getElmNum(k)];
        }
        jetEstimation(i).bind_da_ptr(std::move(daPtrs));
      } else {
        // TODO: implement this
      }
    }
    vertexKindIdxUnfixed++;
  }
}
template class EdgeVector<double>;
template class EdgeVector<float>;
}
