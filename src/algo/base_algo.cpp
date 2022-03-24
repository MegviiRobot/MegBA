/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "algo/base_algo.h"
#include "edge/base_edge.h"

namespace MegBA {
template <typename T>
BaseAlgo<T>::BaseAlgo(const ProblemOption &problemOption,
                      const AlgoOption &algoOption)
    : problemOption{problemOption}, algoOption{algoOption} {}

template <typename T>
void BaseAlgo<T>::solve(const BaseLinearSystem<T> &baseLinearSystem,
                        const EdgeVector<T> &edges, T *xPtr) {
  switch (problemOption.device) {
    case CUDA:
      solveCUDA(baseLinearSystem, edges, xPtr);
      break;
    default:
      throw std::runtime_error("Not implemented");
  }
}

SPECIALIZE_CLASS(BaseAlgo);
}  // namespace MegBA
