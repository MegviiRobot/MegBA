/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "base_linear_system_manager.h"
#include "problem/hessian_entrance.h"

namespace MegBA {
template <typename T>
struct SchurLMLinearSystemManager : public BaseLinearSystemManager<T> {
  explicit SchurLMLinearSystemManager(const ProblemOption &option);

  struct EquationContainer {
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
  };

  struct PositionContainer {
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
  };

  std::vector<T *> deltaXPtrBackup;

  std::vector<EquationContainer> equationContainers;

  std::vector<PositionContainer> positionContainers;

  std::array<int, 2> num{};
  std::array<int, 2> dim{};

  std::vector<std::array<T *, 2>> extractedDiag;

  std::size_t getHessianShape() const override {
    return dim[0] * num[0] + dim[1] * num[1];
  }

  void allocateResourceCUDA();

  void buildPositionContainer(
      const std::vector<SchurHessianEntrance<T>> &schurHessianEntrance,
      const VertexVector<T> &vertexVectorCamera,
      const VertexVector<T> &vertexVectorPoint);

  void processDiag(const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const;

  void backup() const;

  void rollback() const;

  void buildIndex(const BaseProblem<T> &problem) override;

  void solve() const override;

  void applyUpdate(T *xPtr) const override;
};
}