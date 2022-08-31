/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#include "linear_system/implicit_schur_linear_system.h"

#include <stdexcept>

#include "solver/base_solver.h"

namespace MegBA {
template <typename T>
ImplicitSchurLinearSystem<T>::ImplicitSchurLinearSystem(
    const ProblemOption& option, std::unique_ptr<BaseSolver<T>> solver)
    : BaseLinearSystem<T>{option, std::move(solver)},
      implicitEquationContainers{option.deviceUsed.size()} {}

template <typename T>
ImplicitSchurLinearSystem<T>::~ImplicitSchurLinearSystem<T>() {
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
void ImplicitSchurLinearSystem<T>::freeCPU() {}

SPECIALIZE_STRUCT(ImplicitSchurLinearSystem);
}  // namespace MegBA