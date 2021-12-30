/**
* MegBA is Licensed under the Apache License, Version 2.0 (the "License")
*
* Copyright (c) 2021 Megvii Inc. All rights reserved.
*
**/

#pragma once
#include "base_linear_system_manager.h"

namespace MegBA {
template <typename T>
struct SchurLMLinearSystemManager : public BaseLinearSystemManager<T> {
  struct EquationContainer {
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
  };

  struct PositionAndRelationContainer {
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
  };

  std::vector<T *> deltaXPtrBackup{};

  std::vector<EquationContainer> equationContainers;

  std::vector<PositionAndRelationContainer> positionAndRelationContainers;

  std::array<int, 2> num{};
  std::array<int, 2> dim{};

  std::vector<std::array<T *, 2>> extractedDiag;

  std::size_t getHessianShape() const { return dim[0] * num[0] + dim[1] * num[1]; }

  void preSolve(const AlgoStatus &algoStatus) override;

  void buildLinearSystemCUDA(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) override;

  void backup();

  void rollback();

  void applyUpdate(T *xPtr) const override;

  double computeRhoDenominator(JVD<T> &JV, std::vector<T *> &schurDeltaXPtr);
};
}