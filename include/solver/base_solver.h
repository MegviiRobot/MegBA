/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "common.h"

namespace MegBA {
template <typename T>
struct BaseSolver {
  virtual LinearSystemKind linearSystemKind() const {
    return BASE_LINEAR_SYSTEM;
  }

  virtual SolverKind solverKind() const { return BASE_SOLVER; }

  BaseSolver(const ProblemOption &problemOption,
             const SolverOption &solverOption)
      : problemOption{problemOption}, solverOption{solverOption} {}

  virtual ~BaseSolver() = default;

  virtual void solve(const BaseLinearSystem<T> &baseLinearSystem) = 0;

  virtual void solve(const BaseLinearSystem<T> &baseLinearSystem,
                     const EdgeVector<T> &edges,
                     const JVD<T> &jetEstimation) {};

  const ProblemOption &problemOption;
  const SolverOption &solverOption;
};
}  // namespace MegBA
