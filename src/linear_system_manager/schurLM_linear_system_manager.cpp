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
    : LMLinearSystemManager<T>{option},
      equationContainers{option.deviceUsed.size()} {}

template <typename T>
void SchurLMLinearSystemManager<T>::buildIndex(const BaseProblem<T> &problem) {
  const auto &edges = problem.getEdgeVectors().getVertexVectors();
  this->num[0] = problem.getVerticesSets().find(CAMERA)->second.size();
  this->num[1] = problem.getVerticesSets().find(POINT)->second.size();
  this->dim[0] = edges[0][0]->getGradShape();
  this->dim[1] = edges[1][0]->getGradShape();
  const auto &hessianEntrance = problem.getHessianEntrance();
  const auto worldSize = MemoryPool::getWorldSize();
  for (int i = 0; i < worldSize; ++i) {
    equationContainers[i].nnz[0] = hessianEntrance[i].nnzInE;
    equationContainers[i].nnz[1] = hessianEntrance[i].nnzInE;
    equationContainers[i].nnz[2] = this->num[0] * this->dim[0] * this->dim[0];
    equationContainers[i].nnz[3] = this->num[1] * this->dim[1] * this->dim[1];
  }
  allocateResourceCUDA();
}

template struct SchurLMLinearSystemManager<double>;
template struct SchurLMLinearSystemManager<float>;
}
