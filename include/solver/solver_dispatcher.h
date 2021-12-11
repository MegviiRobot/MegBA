/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <memory>

#include "base_solver.h"
#include "problem/base_problem.h"
#include "schur_distributed_pcg_solver.h"

namespace MegBA {
template <typename T>
std::unique_ptr<BaseSolver<T>> dispatchSolver(const BaseProblem<T> &problem) {
  if (problem.option.useSchur) {
    switch (problem.option.solverKind) {
      case PCG:
        return std::unique_ptr<BaseSolver<T>>{new SchurDistributedPCGSolver<T>{
            problem.option, problem.deltaXPtr,
            problem.edges.schurEquationContainer}};
    }
  } else {
    throw std::runtime_error("Not implemented");
  }
}
}  // namespace MegBA
