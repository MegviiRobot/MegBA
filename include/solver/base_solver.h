/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <utility>
#include <vector>

#include "common.h"
#include "problem/base_problem.h"

namespace MegBA {
template <typename T>
struct BaseSolver {
 protected:
  const BaseProblem<T> &problem;

 public:
  explicit BaseSolver(const BaseProblem<T> &problem) : problem(problem) {};

  void solve() {
    switch (problem.getProblemOption().device) {
      case CUDA:
        solveCUDA();
        break;
      default:
        throw std::runtime_error("Not implemented");
    }
  };

  virtual void solveCUDA() = 0;
};
}  // namespace MegBA
