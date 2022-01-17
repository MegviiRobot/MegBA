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
      equationContainers{option.deviceUsed.size()} {}

template <typename T>
void SchurLMLinearSystemManager<T>::buildIndex(const BaseProblem<T> &problem) {
  const auto &edges = problem.getEdgeVectors().getVertexVectors();
  num[0] = problem.getVerticesSets().find(CAMERA)->second.size();
  num[1] = problem.getVerticesSets().find(POINT)->second.size();
  dim[0] = edges[0][0]->getGradShape();
  dim[1] = edges[1][0]->getGradShape();
  const auto &hessianEntrance = problem.getHessianEntrance();
  const auto worldSize = MemoryPool::getWorldSize();
  for (int i = 0; i < worldSize; ++i) {
    equationContainers[i].nnz[0] = hessianEntrance[i].nnzInE;
    equationContainers[i].nnz[1] = hessianEntrance[i].nnzInE;
    equationContainers[i].nnz[2] = num[0] * dim[0] * dim[0];
    equationContainers[i].nnz[3] = num[1] * dim[1] * dim[1];
  }
  allocateResourceCUDA();
}

template struct SchurLMLinearSystemManager<double>;
template struct SchurLMLinearSystemManager<float>;
}
