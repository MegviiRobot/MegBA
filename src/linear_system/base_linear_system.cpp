/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#include "linear_system/base_linear_system.h"
#include "linear_system/schur_LM_linear_system.h"
#include "solver/schur_pcg_solver.h"

namespace MegBA {
template <typename T>
BaseLinearSystem<T>::BaseLinearSystem(const ProblemOption &problemOption,
                                      std::unique_ptr<BaseSolver<T>> solver)
    : problemOption{problemOption},
      deltaXPtr{problemOption.deviceUsed.size()},
      g{problemOption.deviceUsed.size()},
      solver{std::move(solver)} {
  if (this->solver->linearSystemKind() != problemOption.linearSystemKind &&
      this->solver->solverKind() != problemOption.solverKind) {
    throw std::runtime_error("Wrong solver type");
  }
}

template <typename T>
std::size_t BaseLinearSystem<T>::getHessianShape() const {
  return dim[0] * num[0] + dim[1] * num[1];
}

template <typename T>
void BaseLinearSystem<T>::solve() const {
  solver->solve(*this);
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

SPECIALIZE_CLASS(BaseLinearSystem);
}
