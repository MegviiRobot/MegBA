/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "LM_linear_system_manager.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct SchurLMLinearSystemManager : public LMLinearSystemManager<T> {
  explicit SchurLMLinearSystemManager(const ProblemOption &option);

  struct EquationContainer {
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
  };

  std::vector<EquationContainer> equationContainers;

  void allocateResourceCUDA();

  void processDiag(const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const override;

  void backup() const override;

  void rollback() const override;

  void buildIndex(const BaseProblem<T> &problem) override;

  void solve() const override;

  void applyUpdate(T *xPtr) const override;
};
}