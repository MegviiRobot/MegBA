/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "LM_linear_system.h"

namespace MegBA {
template <typename T>
struct SchurLinearSystem : virtual public BaseLinearSystem<T> {
  LinearSystemKind linearSystemKind() const override { return SCHUR; }

  SchurLinearSystem() = delete;

  virtual ~SchurLinearSystem();

  struct EquationContainer {
    std::array<int *, 2> csrRowPtr{nullptr, nullptr};
    std::array<T *, 4> csrVal{nullptr, nullptr, nullptr, nullptr};
    std::array<int *, 2> csrColInd{nullptr, nullptr};
    std::array<std::size_t, 4> nnz{0, 0, 0, 0};
  };

  std::vector<EquationContainer> equationContainers;

 protected:
  explicit SchurLinearSystem(const ProblemOption &option,
                             std::unique_ptr<BaseSolver<T>> solver);

 private:
  void freeCPU();

  void freeCUDA();
};
}  // namespace MegBA
