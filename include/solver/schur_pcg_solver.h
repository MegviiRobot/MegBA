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
struct SchurPCGSolver : public BaseSolver<T> {
  void solve(const BaseLinearSystem<T>& baseLinearSystem) final;
};
}  // namespace MegBA
