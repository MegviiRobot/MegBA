/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/schur_linear_system.h"

namespace MegBA {
template <typename T>
SchurLinearSystem<T>::SchurLinearSystem(const ProblemOption& option,
                                        std::unique_ptr<BaseSolver<T>> solver)
    : BaseLinearSystem<T>{option, std::move(solver)},
      equationContainers{option.deviceUsed.size()} {}

template <typename T>
SchurLinearSystem<T>::~SchurLinearSystem<T>() {
  switch (this->problemOption.device) {
    case CUDA:
      freeCUDA();
      break;
    case CPU:
      freeCPU();
      break;
  }
}
template <typename T>
void SchurLinearSystem<T>::freeCPU() {}

template class SchurLinearSystem<double>;
template class SchurLinearSystem<float>;
}  // namespace MegBA
