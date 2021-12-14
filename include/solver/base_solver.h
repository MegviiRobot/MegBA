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

namespace MegBA {
template <typename T>
class BaseSolver {
 protected:
  const ProblemOption option;
  const std::vector<T *> &deltaXPtr;

 public:
  explicit BaseSolver(const ProblemOption &option,
                      const std::vector<T *> &deltaXPtr)
      : option(option), deltaXPtr(deltaXPtr){};

  void solve() {
    switch (option.device) {
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
