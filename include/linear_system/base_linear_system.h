/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <vector>
#include "problem/base_problem.h"
#include "algo/base_algo.h"

namespace MegBA {
template <typename T>
struct BaseLinearSystem {
  static std::unique_ptr<BaseLinearSystem<T>> dispatch(const BaseProblem<T> *problem);

  const SolverOption &solverOption;
  std::vector<T *> deltaXPtr;
  std::vector<T *> g;

  std::array<int, 2> num{};
  std::array<int, 2> dim{};

  std::size_t getHessianShape() const {
    return dim[0] * num[0] + dim[1] * num[1];
  }

  virtual void solve() const = 0;

  virtual void buildIndex(const BaseProblem<T> &problem) = 0;

  virtual void applyUpdate(T *xPtr) const = 0;

 protected:
  explicit BaseLinearSystem(const ProblemOption &option);
};
}
