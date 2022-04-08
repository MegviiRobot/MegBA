/**
 * MegBA is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2021 Megvii Inc. All rights reserved.
 *
 **/

#pragma once
#include "base_linear_system.h"

namespace MegBA {
template <typename T>
struct LMLinearSystem : virtual public BaseLinearSystem<T> {
  AlgoKind algoKind() const override { return LM; }

  LMLinearSystem() = delete;

  virtual ~LMLinearSystem();

  std::vector<T *> deltaXPtrBackup;

  std::vector<T *> gBackup;

  std::vector<std::array<T *, 2>> extractedDiag;

  virtual void processDiag(
      const AlgoStatus::AlgoStatusLM &lmAlgoStatus) const = 0;

  virtual void backup() const = 0;

  virtual void rollback() const = 0;

 protected:
  explicit LMLinearSystem(const ProblemOption &option,
                          std::unique_ptr<BaseSolver<T>> solver);

 private:
  void freeCPU();

  void freeCUDA();
};
}  // namespace MegBA