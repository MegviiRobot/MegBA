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
class SchurDistributedPCGSolver : public BaseSolver<T> {
 public:
  explicit SchurDistributedPCGSolver(const BaseProblem<T> &problem)
      : BaseSolver<T>(problem) {};

  void solveCUDA() final;
};
}  // namespace MegBA
