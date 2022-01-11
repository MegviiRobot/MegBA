/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "base_algo.h"

namespace MegBA {
template <typename T>
class LMAlgo : public BaseAlgo<T> {
 public:
  explicit LMAlgo(const AlgoOption &option);

  void solveCUDA(const BaseLinearSystemManager<T> &baseLinearSystemManager,
                 const EdgeVector<T> &edges,
                 T *xPtr) override;
};
}