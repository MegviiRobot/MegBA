/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system_manager/schurLM_linear_system_manager.h"
#include <omp.h>

namespace MegBA {
template <typename T>
SchurLMLinearSystemManager<T>::SchurLMLinearSystemManager(
    const ProblemOption &option)
    : BaseLinearSystemManager<T>{option},
      deltaXPtrBackup{option.deviceUsed.size()},
      equationContainers{option.deviceUsed.size()},
      positionContainers{option.deviceUsed.size()} {}

template <typename T>
void SchurLMLinearSystemManager<T>::buildPositionContainer(
    const std::vector<SchurHessianEntrance<T>> &schurHessianEntrance,
    const VertexVector<T> &vertexVectorCamera,
    const VertexVector<T> &vertexVectorPoint) {
  const auto worldSize = MemoryPool::getWorldSize();
  positionContainers.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    positionContainers[i].absolutePositionCamera =
        (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
    positionContainers[i].absolutePositionPoint =
        (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
    positionContainers[i].relativePositionCamera =
        (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
    positionContainers[i].relativePositionPoint =
        (int *)malloc(MemoryPool::getItemNum(i) * sizeof(int));
  }

  std::size_t totalVertexIdx{0};
  for (int i = 0; i < worldSize; ++i) {
    const auto &schurHEntranceCamera = schurHessianEntrance[i].ra[0];
    const auto &schurHEntrancePoint = schurHessianEntrance[i].ra[1];
    omp_set_num_threads(16);
#pragma omp parallel for
    for (int j = 0; j < schurHessianEntrance[i].counter; ++j) {
      {
        const auto &row =
            schurHEntrancePoint[vertexVectorPoint[totalVertexIdx + j]
                                    ->absolutePosition];
        positionContainers[i].relativePositionCamera[j] = std::distance(
            row.begin(),
            std::lower_bound(row.begin(), row.end(),
                             vertexVectorCamera[totalVertexIdx + j]));
        positionContainers[i].absolutePositionCamera[j] =
            vertexVectorCamera[totalVertexIdx + j]->absolutePosition;
      }
      {
        const auto &row =
            schurHEntranceCamera[vertexVectorCamera[totalVertexIdx + j]
                                     ->absolutePosition];
        positionContainers[i].relativePositionPoint[j] = std::distance(
            row.begin(),
            std::lower_bound(row.begin(), row.end(),
                             vertexVectorPoint[totalVertexIdx + j]));
        positionContainers[i].absolutePositionPoint[j] =
            vertexVectorPoint[totalVertexIdx + j]->absolutePosition;
      }
    }
    totalVertexIdx += schurHessianEntrance[i].counter;
    // fill csrRowPtr_. next row's csrRowPtr_ = this row's csrRowPtr_ + this
    // row's non-zero element number.

    equationContainers[i].nnz[0] = schurHessianEntrance[i].nnzInE;
    equationContainers[i].nnz[1] = schurHessianEntrance[i].nnzInE;
    equationContainers[i].nnz[2] = num[0] * dim[0] * dim[0];
    equationContainers[i].nnz[3] = num[1] * dim[1] * dim[1];
  }
}

template <typename T>
void SchurLMLinearSystemManager<T>::buildIndex(const BaseProblem<T> &problem) {
  const auto &edges = problem.getEdges().getEdges();
  num[0] = problem.getVerticesSets().find(CAMERA)->second.size();
  num[1] = problem.getVerticesSets().find(POINT)->second.size();
  dim[0] = edges[0][0]->getGradShape();
  dim[1] = edges[1][0]->getGradShape();
  buildPositionContainer(problem.getSchurHessianEntrance(),
                         edges[0],
                         edges[1]);
  allocateResourceCUDA();
}

template struct SchurLMLinearSystemManager<double>;
template struct SchurLMLinearSystemManager<float>;
}
