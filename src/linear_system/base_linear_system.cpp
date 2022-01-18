/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system/base_linear_system.h"
#include "linear_system/schurLM_linear_system.h"

namespace MegBA {
template <typename T>
BaseLinearSystem<T>::BaseLinearSystem(const ProblemOption &option)
    : solverOption{option.solverOption},
      deltaXPtr{option.deviceUsed.size()},
      g{option.deviceUsed.size()} {}

template <typename T>
std::unique_ptr<BaseLinearSystem<T>> BaseLinearSystem<T>::dispatch(
    const BaseProblem<T> *problem) {
  const ProblemOption &option = problem->getProblemOption();
  if (option.useSchur) {
    switch (option.algoKind) {
      case LM:
        return std::unique_ptr<BaseLinearSystem<T>>{
            new SchurLMLinearSystem<T>{option}};
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
}

template class BaseLinearSystem<double>;
template class BaseLinearSystem<float>;
}
