/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <vector>
#include "algo/base_algo.h"

namespace MegBA {
template <typename T>
struct BaseLinearSystemManager {
  explicit BaseLinearSystemManager(const ProblemOption &option)
      : solverOption{option.solverOption},
        deltaXPtr{option.deviceUsed.size()} {}

  const SolverOption &solverOption;
  std::vector<T *> deltaXPtr;

  virtual std::size_t getHessianShape() const = 0;

  virtual void solve() const = 0;

  virtual void buildIndex(const BaseProblem<T> &problem) = 0;

  virtual void applyUpdate(T *xPtr) const = 0;
};
}
