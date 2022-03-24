/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "edge/base_edge.h"
#include "solver/base_solver.h"

namespace MegBA {
template <typename T>
struct SchurSolver : virtual public BaseSolver<T> {
  LinearSystemKind linearSystemKind() const override { return SCHUR; }

  SchurSolver(const ProblemOption &problemOption,
              const SolverOption &solverOption)
      : BaseSolver<T>{problemOption, solverOption} {};
};
}  // namespace MegBA
