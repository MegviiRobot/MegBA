/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/LM_linear_system.h"

namespace MegBA {
template <typename T>
LMLinearSystem<T>::LMLinearSystem(const ProblemOption& option,
                                  std::unique_ptr<BaseSolver<T>> solver)
    : BaseLinearSystem<T>{option, std::move(solver)},
      deltaXPtrBackup{option.deviceUsed.size()},
      gBackup{option.deviceUsed.size()},
      extractedDiag{option.deviceUsed.size()} {}

template <typename T>
LMLinearSystem<T>::~LMLinearSystem<T>() {
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
void LMLinearSystem<T>::freeCPU() {}

template class LMLinearSystem<double>;
template class LMLinearSystem<float>;
}  // namespace MegBA
