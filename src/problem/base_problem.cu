/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/base_linear_system.h"
#include "problem/base_problem.h"

namespace MegBA {
template <typename T>
void BaseProblem<T>::deallocateResourceCUDA() {
  for (int i = 0; i < MemoryPool::getWorldSize(); ++i) {
    cudaSetDevice(i);
    cudaFree(xPtr[i]);
  }
  xPtr.clear();
}

template <typename T>
void BaseProblem<T>::allocateResourceCUDA() {
  const auto worldSize = MemoryPool::getWorldSize();
  xPtr.resize(worldSize);
  for (int i = 0; i < worldSize; ++i) {
    cudaSetDevice(i);
    cudaMalloc(&xPtr[i], linearSystem->getHessianShape() * sizeof(T));
  }
}

template class BaseProblem<double>;
template class BaseProblem<float>;
}  // namespace MegBA
