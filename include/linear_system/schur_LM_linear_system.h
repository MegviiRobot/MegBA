/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "schur_linear_system.h"
#include "LM_linear_system.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct SchurLMLinearSystem : public SchurLinearSystem<T>, public LMLinearSystem<T> {
  explicit SchurLMLinearSystem(
      const ProblemOption &option)
      : SchurLinearSystem<T>{option}, LMLinearSystem<T>{option}, BaseLinearSystem<T>{option} {}

  void allocateResourceCUDA();

  void processDiag(const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const override;

  void backup() const override;

  void rollback() const override;

  void buildIndex(const BaseProblem<T> &problem) override;

  void applyUpdate(T *xPtr) const override;
};
}