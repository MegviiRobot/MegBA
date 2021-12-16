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
class BaseLinearSystemManager {
  std::vector<T *> deltaXPtr{};

  virtual void buildLinearSystem(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) = 0;

  virtual void buildLinearSystemCUDA(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) = 0;

  virtual void preProcess(const AlgoStatus &algoStatus) {}
};
}
