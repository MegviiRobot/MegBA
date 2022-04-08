/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include <memory>

#include "common.h"

namespace MegBA {
template <typename T>
struct BaseAlgo {
  virtual AlgoKind algoKind() const { return BASE_ALGO; }

  void solve(const BaseLinearSystem<T> &baseLinearSystem,
             const EdgeVector<T> &edges, T *xPtr);

  virtual void solveCUDA(const BaseLinearSystem<T> &baseLinearSystem,
                         const EdgeVector<T> &edges, T *xPtr) = 0;

 protected:
  explicit BaseAlgo(const ProblemOption &problemOption,
                    const AlgoOption &algoOption);

  const ProblemOption &problemOption;
  const AlgoOption &algoOption;

  AlgoStatus algoStatus{};
};
}  // namespace MegBA
