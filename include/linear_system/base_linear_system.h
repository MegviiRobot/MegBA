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
#include "solver/base_solver.h"

namespace MegBA {
template <typename T>
struct BaseLinearSystem {
  BaseLinearSystem() = delete;

  virtual ~BaseLinearSystem();

  const ProblemOption &problemOption;
  std::vector<T *> deltaXPtr;
  std::vector<T *> g;

  std::array<int, 2> num{};
  std::array<int, 2> dim{};

  std::unique_ptr<BaseSolver<T>> solver;

  std::size_t getHessianShape() const;

  void solve() const;

  virtual void buildIndex(const BaseProblem<T> &problem) = 0;

  virtual void applyUpdate(T *xPtr) const = 0;

 protected:
  explicit BaseLinearSystem(const ProblemOption &problemOption, std::unique_ptr<BaseSolver<T>> solver);

 private:
  void freeCPU();

  void freeCUDA();
};
}
