/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system/base_linear_system.h"
#include "linear_system/schur_LM_linear_system.h"

namespace MegBA {
template <typename T>
BaseLinearSystem<T>::BaseLinearSystem(const ProblemOption &problemOption)
    : problemOption{problemOption},
      deltaXPtr{problemOption.deviceUsed.size()},
      g{problemOption.deviceUsed.size()} {}

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

template <typename T>
std::size_t BaseLinearSystem<T>::getHessianShape() const {
  return dim[0] * num[0] + dim[1] * num[1];
}

template <typename T>
BaseLinearSystem<T>::~BaseLinearSystem() {
  switch (problemOption.device) {
    case CUDA:
      freeCUDA();
      break;
    case CPU:
      freeCPU();
      break;
  }
}
template <typename T>
void BaseLinearSystem<T>::freeCPU() {
  // TODO (Jie): implement this
}

template class BaseLinearSystem<double>;
template class BaseLinearSystem<float>;
}
