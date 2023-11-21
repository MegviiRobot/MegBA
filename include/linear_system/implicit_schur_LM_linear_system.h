/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "LM_linear_system.h"
#include "implicit_schur_linear_system.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct ImplicitSchurLMLinearSystem : public ImplicitSchurLinearSystem<T>,
                                     public LMLinearSystem<T> {
  explicit ImplicitSchurLMLinearSystem(const ProblemOption &option,
                                       std::unique_ptr<BaseSolver<T>> solver);

  void allocateResourceCUDA();

  void processDiag(const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const override;

  void backup() const override;

  void rollback() const override;

  void buildIndex(const BaseProblem<T> &problem) override;

  void applyUpdate(T *xPtr) const override;
};
}  // namespace MegBA