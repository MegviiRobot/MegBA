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
  std::vector<T *> deltaXPtr{};

  void buildLinearSystem(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) {
      buildLinearSystemCUDA(jetEstimation, jetInformation);
  };

  virtual void preSolve(const AlgoStatus &algoStatus) {}

  virtual void buildLinearSystemCUDA(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) = 0;

  virtual void postSolve(const AlgoStatus &algoStatus) {}

  virtual void applyUpdate(T *xPtr) const = 0;
};
}
