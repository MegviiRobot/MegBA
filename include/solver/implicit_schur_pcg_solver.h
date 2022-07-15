/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "edge/base_edge.h"
#include "pcg_solver.h"
#include "schur_solver.h"

namespace MegBA {
template <typename T>
struct ImplicitSchurPCGSolver : public PCGSolver<T>, public SchurSolver<T> {
  ImplicitSchurPCGSolver(const ProblemOption &problemOption,
                const SolverOption &solverOption);

 void solve(const BaseLinearSystem<T> &baseLinearSystem){};

 void solve(const BaseLinearSystem<T> &baseLinearSystem, const EdgeVector<T> &edges, const JVD<T> &jetEstimation);
};
}  // namespace MegBA

