/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system/schurLM_linear_system.h"
#include <omp.h>

namespace MegBA {
template <typename T>
SchurLMLinearSystem<T>::SchurLMLinearSystem(
    const ProblemOption &option)
    : LMLinearSystem<T>{option},
      equationContainers{option.deviceUsed.size()} {}

template <typename T>
void SchurLMLinearSystem<T>::buildIndex(const BaseProblem<T> &problem) {
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

template struct SchurLMLinearSystem<double>;
template struct SchurLMLinearSystem<float>;
}
