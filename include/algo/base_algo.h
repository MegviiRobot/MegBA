/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include <memory>
#include "common.h"
#include "problem/base_problem.h"

namespace MegBA {
struct AlgoStatus {
  struct AlgoStatusLM {
    double region;
    bool recoverDiag{false};
  } algoStatusLM;
};

template <typename T>
class BaseAlgo {
 protected:
  const AlgoOption &algoOption;
  AlgoStatus algoStatus{};

 public:
  explicit BaseAlgo(const AlgoOption &algoOption) : algoOption{algoOption} {}

  void solve(const BaseLinearSystem<T> &baseLinearSystem,
             const EdgeVector<T> &edges, T *xPtr) {
    switch (algoOption.device) {
      case CUDA:
        solveCUDA(baseLinearSystem, edges, xPtr);
        break;
      default:
        throw std::runtime_error("Not implemented");
    }
  };

  virtual void solveCUDA(
      const BaseLinearSystem<T> &baseLinearSystem,
      const EdgeVector<T> &edges, T *xPtr) = 0;
};
}
