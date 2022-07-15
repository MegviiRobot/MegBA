/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <memory>
#include <vector>
#include <array>
#include "common.h"
#include "vertex/base_vertex.h"

namespace MegBA {
template <typename T>
struct BaseLinearSystem {
  virtual AlgoKind algoKind() const { return BASE_ALGO; }

  virtual LinearSystemKind linearSystemKind() const {
    return BASE_LINEAR_SYSTEM;
  }

  virtual ComputeKind computeKind() const {
    return EXPLICIT;
  }

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

  virtual void solve(const EdgeVector<T> &edges, const JVD<T> &jetEstimation) const {};

  virtual void buildIndex(const BaseProblem<T> &problem) = 0;

  virtual void applyUpdate(T *xPtr) const = 0;

 protected:
  explicit BaseLinearSystem(const ProblemOption &problemOption,
                            std::unique_ptr<BaseSolver<T>> solver);

 private:
  void freeCPU();

  void freeCUDA();
};
}  // namespace MegBA
