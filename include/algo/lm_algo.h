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
struct LMAlgo : public BaseAlgo<T> {
  AlgoKind algoKind() const override { return LM; }

  explicit LMAlgo(const ProblemOption &problemOption,
                  const AlgoOption &algoOption);

  void solveCUDA(const BaseLinearSystem<T> &baseLinearSystem,
                 const EdgeVector<T> &edges, T *xPtr) override;
};
}  // namespace MegBA