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
class SchurLMLinearSystemManager : public BaseLinearSystemManager<T> {
  struct EquationContainer {
    explicit EquationContainer(const Device &device) : device(device) {}

    ~EquationContainer() { clear(); }

    void clear();

    void clearCUDA();
    const Device &device;
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    T *g{nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
    std::array<int, 2> dim{0, 0};
  };

  struct PositionAndRelationContainer {
    explicit PositionAndRelationContainer(const Device &device)
        : device(device) {}

    ~PositionAndRelationContainer() { clear(); }

    void clear();

    void clearCUDA();

    const Device &device;
    int *relativePositionCamera{nullptr}, *relativePositionPoint{nullptr};
    int *absolutePositionCamera{nullptr}, *absolutePositionPoint{nullptr};
    int *connectionNumPoint{nullptr};
  };

  std::vector<EquationContainer> equationContainers;

  std::vector<PositionAndRelationContainer> positionAndRelationContainers;

  std::vector<std::array<T *, 2>> extractedDiag;

  void buildLinearSystem(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) override;

  void buildLinearSystemCUDA(const JVD<T> &jetEstimation, const JVD<T> &jetInformation) override;

  void preProcess(const AlgoStatus &algoStatus) override;
};
}