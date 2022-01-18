/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "algo/base_algo.h"
#include "algo/lm_algo.h"
#include "problem/base_problem.h"

namespace MegBA {
template <typename T>
BaseAlgo<T>::BaseAlgo(const AlgoOption &algoOption) : algoOption{algoOption} {}

template <typename T>
std::unique_ptr<BaseAlgo<T>> BaseAlgo<T>::dispatch(
    const BaseProblem<T> *problem) {
  const ProblemOption &option = problem->getProblemOption();
  if (option.useSchur) {
    switch (option.algoKind) {
      case LM:
        return std::unique_ptr<BaseAlgo<T>>{new LMAlgo<T>{option.algoOption}};
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
}

template <typename T>
void BaseAlgo<T>::solve(const BaseLinearSystem<T> &baseLinearSystem,
                        const EdgeVector<T> &edges, T *xPtr) {
  switch (algoOption.device) {
    case CUDA:
      solveCUDA(baseLinearSystem, edges, xPtr);
      break;
    default:
      throw std::runtime_error("Not implemented");
  }
}

template class BaseAlgo<double>;
template class BaseAlgo<float>;
}