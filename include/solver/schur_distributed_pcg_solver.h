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
  using SchurEquationContainer = typename EdgeVector<T>::SchurEquationContainer;
  const std::vector<SchurEquationContainer> &schurEquationContainers;

 public:
  explicit SchurDistributedPCGSolver(
      const ProblemOption &option, const std::vector<T *> &deltaXPtr,
      const std::vector<SchurEquationContainer> &schurEquationContainers)
      : BaseSolver<T>(option, deltaXPtr),
        schurEquationContainers(schurEquationContainers){};

  void solveCUDA() final;
};
}  // namespace MegBA
